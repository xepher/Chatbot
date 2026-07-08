# encoding=utf-8
import json
import os
import torch
import sys

# ==========================================
# 1. 运行设备配置 (CPU / GPU)
# ==========================================
# 自动检测当前是否有可用的 NVIDIA CUDA 显卡。如果有，则使用 GPU 训练以加速；否则回退到 CPU 运行。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 语料下载地址与本地路径配置
# ==========================================
# 小黄鸡语料（xiaohuangji50w）的下载链接和解压后的本地文件路径
xhj_corpus_url = 'https://github.com/candlewill/Dialog_Corpus/raw/master/xiaohuangji50w_nofenci.conv.zip'
xhj_corpus_loc = 'data/xiaohuangji50w_nofenci.conv'

# PTT中文问答语料的下载链接和下载后的本地文件路径
ptt_corpus_url = 'https://github.com/zake7749/Gossiping-Chinese-Corpus/raw/master/data/Gossiping-QA-Dataset.txt'
ptt_corpus_loc = 'data/Gossiping-QA-Dataset.txt'

# ==========================================
# 3. 命令行参数解析与数据保存路径动态映射
# ==========================================
# 自动拦截并解析命令行中的 --corpus 或 -c 参数。
# 比如运行：python pre_process.py --corpus ptt 时，自动切换到 ptt 语料对应的输出文件。
corpus = 'xhj'
for idx, arg in enumerate(sys.argv):
    if arg in ('--corpus', '-c') and idx + 1 < len(sys.argv):
        corpus = sys.argv[idx + 1]
        break

# 根据当前选择的语料，为词表 (WORDMAP) 和处理后的样本 (samples.json) 映射不同的文件名，防止相互覆盖。
if corpus == 'ptt':
    wordmap_loc = 'data/WORDMAP_ptt.json'
    samples_loc = 'data/samples_ptt.json'
else:
    wordmap_loc = 'data/WORDMAP.json'
    samples_loc = 'data/samples.json'


# ==========================================
# 4. 训练与优化器超参数配置
# ==========================================
clip = 50.0                  # 梯度裁剪阀值。防止 RNN 训练过程中梯度爆炸（Gradient Explosion）。
teacher_forcing_ratio = 1.0  # 教师迫导比例。1.0 表示在训练的每一步，Decoder 都直接使用真实词作为下一步的输入。
learning_rate = 0.0001       # 学习率。控制参数更新的步伐大小。
decoder_learning_ratio = 5.0 # 解码器的学习率倍数。Luong 论文中提到解码器可以以稍大一些的学习率运行。
n_iteration = 4000           # 训练迭代的总批次数（如果是按 iteration 算）。
print_every = 100            # 每隔多少个 Batch 打印一次训练日志。
save_every = 1               # 每隔多少个 Epoch/Batch 保存一次模型 Checkpoint。
workers = 1                  # 数据加载的多线程数（Windows 下通常设为 0 或 1 防止多线程冲突）。
max_len = 10                 # 句子截断长度。为了提高计算效率，丢弃或截断长度超过 10 个词的句子。
min_word_freq = 3            # 词频过滤阀值。在构建词表时，只保留在全部语料中出现次数大于 3 的词，过滤低频噪音词。
save_dir = 'models'          # 模型保存的目录。

# ==========================================
# 5. 模型结构超参数配置
# ==========================================
model_name = 'cb_model'      # 模型的名称。
attn_model = 'dot'           # 注意力计算方式。Luong Attention 支持: 'dot'（点积）、'general'、'concat'。
start_epoch = 0              # 训练起始 Epoch，若从中断点恢复，则为 Checkpoint 记录的值。
epochs = 120                 # 总的训练轮数 (Epoch)。
hidden_size = 500            # 隐藏层状态维度大小（GRU的隐状态通道数，以及 Word Embedding 的维度）。
encoder_n_layers = 2         # Encoder (编码器) 的 GRU 堆叠层数。
decoder_n_layers = 2         # Decoder (解码器) 的 GRU 堆叠层数。
dropout = 0.1                # Dropout 比例。随机失活 10% 的神经元，用于防止过拟合。
batch_size = 64              # 批大小。每个训练步骤同时输入模型的句子数量。

# ==========================================
# 6. 中断恢复 (Checkpoint) 配置
# ==========================================
loadFilename = None          # 如果需要加载已有的模型继续训练，将此路径修改为 Checkpoint 文件路径，否则为 None。

# ==========================================
# 7. 特殊 Token 定义
# ==========================================
# 神经网络无法直接处理变长序列，需要这 4 个特殊的控制字符来协助序列处理：
PAD_token = 0  # <pad> 填充 Token：把短句子补充对齐到 Batch 内最大长度。
SOS_token = 1  # <start> / SOS 开始 Token：Decoder 预测第一个词时，作为初始的输入符号。
EOS_token = 2  # <end> / EOS 结束 Token：Decoder 预测出该 Token 时表示句子生成结束。
UNK_token = 3  # <unk> 未知词 Token：遇到词表中没有的词（例如罕见词或错别字）时，以此代替。


# ==========================================
# 8. 词表类定义 (Voc)
# ==========================================
class Voc:
    """
    词表管理类：负责将词语文本与数字 ID 相互映射。
    """
    def __init__(self, filename):
        # 从导出的 JSON 词表中加载词图
        word_map = json.load(open(filename, 'r'))
        self.word2index = word_map                                    # 词 -> ID 映射词表
        self.index2word = {v: k for k, v in word_map.items()}         # ID -> 词 映射表
        self.num_words = len(word_map)                               # 词表中唯一的词汇总数


# 如果本地已生成词表，则自动实例化一个全局的 voc 词表实例，方便其他脚本共享
if os.path.isfile(wordmap_loc):
    voc = Voc(wordmap_loc)
