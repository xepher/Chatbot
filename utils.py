# encoding=utf-8
import datetime
import os
import random
import re
import time
import unicodedata
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="jieba")

import jieba
from torch import nn
from config import *


def ensure_folder(folder):
    """
    检查文件夹是否存在，若不存在则创建。
    """
    if not os.path.exists(folder):
        os.makedirs(folder)


def maskNLLLoss(inp, target, mask):
    """
    掩码负对数似然损失函数 (Masked Negative Log-Likelihood Loss)。
    
    由于我们的批次数据为了对齐，在短句子尾部填充了大量的 <pad> (0) 字符，
    这些无效的填充字符不需要计入损失更新模型。本损失函数通过 Mask 机制实现了忽略填充位置 Loss 的计算。
    
    参数解释 (适合新手学习):
    - inp: Decoder 输出的概率分布，形状为 (batch_size, vocab_size)。它是经过 Softmax 后的全词表词语概率。
    - target: 当前时刻目标答句的真实单词 ID 张量，形状为 (batch_size)。
    - mask: 答句对应时间步的掩码张量，形状为 (batch_size)。1 代表是真实有效的词，0 代表是填充符 <pad>。
    
    计算步骤详解:
    1. target.view(-1, 1):
       将一维的真实词 ID 张量转为 (batch_size, 1) 的纵列矩阵，以便使用 gather 算子。
    2. torch.gather(inp, 1, target.view(-1, 1)):
       利用 gather 算子，在 inp 概率分布矩阵的每一行中，将对应真实词索引位置上的预测概率挑出来。形状变为 (batch_size, 1)。
    3. torch.log(...):
       对概率值取自然对数。概率处于 0~1 之间，因此取对数后为负数。
    4. crossEntropy = -torch.log(...):
       取负对数。概率越大，损失值越趋向 0；概率越小，损失值越大。
    5. crossEntropy.masked_select(mask.bool()):
       利用 Bool 掩码进行条件选择过滤。仅保留 mask == 1（即非填充的真实单词）位置上的 Loss，过滤丢弃所有 padding 造成的 0 损耗。
    6. loss.mean():
       求出当前批次中所有有效字符的平均损失。
    """
    nTotal = mask.sum()
    # 提取目标位置的预测概率并计算其负对数似然值
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)))
    # 根据掩码矩阵过滤无用填充位置的 Loss，并计算剩余有效项的均值
    loss = crossEntropy.masked_select(mask.bool()).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


class AverageMeter(object):
    """
    用来在训练/测试过程中实时累计和计算某种指标的瞬时值、平均值、总和以及运行总步数的计算类工具。
    比如用来统计一个 Epoch 内各个 Batch 的平均 Loss。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        重置所有统计指标。
        """
        self.val = 0    # 当前 Batch 瞬时指标值
        self.avg = 0    # 运行至今的全局平均值
        self.sum = 0    # 运行至今的指标值总和
        self.count = 0  # 累计处理的样本或批次总数

    def update(self, val, n=1):
        """
        输入新的指标值，更新内部统计值。
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class GreedySearchDecoder(nn.Module):
    """
    自回归贪心搜索解码器 (Greedy Search Decoder)。
    在推理或人机交互阶段，它作为一个包裹器模块：
    1. 前向传播编码问句，取得上下文特征。
    2. 在解码阶段，每次迭代选取当前词表概率分布中最大值（即最可能的词）输出，
       并把它作为下一次迭代循环的输入，直到达到最大解码长度上限。
    """
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        """
        推理前向解码。
        
        参数:
        - input_seq: 输入的问句 ID 序列张量，形状为 (seq_len_in, batch_size=1)
        - input_length: 问句的实际长度，一维张量，形状为 (batch_size=1)
        - max_length: 本句允许生成的回答单词的最大长度限制
        
        返回:
        - all_tokens: 解码出的单词 ID 序列张量，形状为 (max_length)
        - all_scores: 解码出的单词对应的 Softmax 置信度打分，形状为 (max_length)
        """
        # 1. 运行 Encoder，得到编码输出及最终的语义隐藏状态
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        
        # 2. 将编码器最后的隐状态用作解码器初始的隐藏状态
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        
        # 3. 初始化 Decoder 的第 0 步输入单词为开始控制符 SOS_token (ID 1)
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        
        # 4. 初始化用于拼接存储解码结果的容器
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        
        # 5. 循环解码 max_length 步
        for _ in range(max_length):
            # 前向计算一个步长词的概率
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            
            # 使用 torch.max() 在 dim=1（词表维度）上获取概率最大项及其索引 ID
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            
            # 记录当前步预测的 Token 和概率分数
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            
            # 用 unsqueeze(0) 将当前步预测的 Token ID 重塑为适合下次输入的 (1, 1) 的形状
            decoder_input = torch.unsqueeze(decoder_input, 0)
            
        return all_tokens, all_scores


def evaluate(indexes_batch, searcher, voc, max_length=max_len):
    """
    评估预测接口。传入问句 ID 列表，运行搜索解码，并转译回文本字符。
    """
    # 建立句子实际长度 Tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # 维度转置以适配 Encoder 接口的批次第一维排列 (seq_len, batch_size)
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    
    # 运行贪心解码
    tokens, scores = searcher(input_batch, lengths, max_length)
    # 将输出数字 ID 转译回具体词汇
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def unicodeToAscii(s):
    """
    将 Unicode 格式的中英文字符进行 ASCII 标准规范化处理（用于英文字符去除重音标点）。
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    """
    对普通英文文本字符串进行小写转换、去标点符号和空格清理规整操作（主要供语料清洗用）。
    """
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def indexesFromSentence(voc, sentence):
    """
    切词并将句子中的中文汉字/词语翻译为词表中关联的数值 ID。末尾自动追加句子结束符 EOS_token。
    """
    return [voc.word2index[word] for word in jieba.cut(sentence)] + [EOS_token]


def pick_n_valid_sentences(n):
    """
    随机从验证集中抽选 n 句话，在训练过程中用来作为对话问句让模型回答，直观考察对话水平的变化。
    """
    samples = json.load(open(samples_loc, 'r'))
    num_train = int(len(samples) * 0.95)
    samples = samples[num_train:]
    samples = random.sample(samples, n)
    return [' '.join([voc.index2word[token] for token in sample['input'] if token != EOS_token]) for sample in samples]


def timestamp():
    """
    生成标准格式的当前时间戳字符串，用于打印日志。
    """
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')


def find_checkpoint(directory, mode='latest'):
    """
    在模型目录下寻找最优或者最新的 Checkpoint 权重包，以便在恢复训练时能实现自动断点续训。
    
    参数:
    - directory: 模型所在文件夹目录
    - mode:
      - 'latest': 寻找 Epoch 值最大的那个文件（最新训练）
      - 'best': 寻找文件名中记录验证集损失 (val_loss) 最低的文件（最优模型）
      
    返回:
    - 匹配的最优/最新 Checkpoint 绝对路径，如果不存在则返回 None。
    """
    if not os.path.exists(directory):
        return None

    # 获取所有 checkpoint_xxx.tar 样式的文件
    files = [f for f in os.listdir(directory) if f.startswith('checkpoint_') and f.endswith('.tar')]
    if not files:
        return None

    parsed_files = []
    for f in files:
        # 裁剪出中间的 epoch 和 loss 部分进行数值转换。例如: checkpoint_12_3.245.tar -> '12', '3.245'
        name_parts = f[11:-4].split('_')
        if len(name_parts) == 2:
            try:
                epoch = int(name_parts[0])
                val_loss = float(name_parts[1])
                parsed_files.append((f, epoch, val_loss))
            except ValueError:
                continue

    if not parsed_files:
        return None

    # 排序选优
    if mode == 'best':
        # 按 val_loss 升序排列，挑出最小 loss 的 checkpoint
        parsed_files.sort(key=lambda x: x[2])
    else:
        # 按 epoch 降序排列，挑出最大 epoch 迭代的 checkpoint
        parsed_files.sort(key=lambda x: x[1], reverse=True)

    return os.path.join(directory, parsed_files[0][0])
