# encoding=utf-8
from collections import Counter
import jieba
from tqdm import tqdm
from config import *


def encode_text(word_map, c):
    """
    将分词后的文本词语列表，转换为词表中对应的 ID 列表，并在末尾附加表示句子结束的 <end> Token。
    
    参数:
    - word_map: 词表映射字典 (词 -> ID)
    - c: 句子分词后的词语列表, 例如 ['我', '爱', '学习']
    
    返回:
    - 带有结束控制符的数字 ID 列表, 例如 [14, 256, 99, 2]
    """
    # 查找词语对应的 ID，如果该词语未包含在词表（word_map）中，则用全局定义的 <unk> (ID 3) 代替。
    # 最后在整个序列最后追加 word_map['<end>'] (即 EOS_token, ID 2) 作为结束符。
    return [word_map.get(word, word_map['<unk>']) for word in c] + [word_map['<end>']]


def build_wordmap(sentences):
    """
    根据句子文本列表统计词频，过滤掉低频词，然后构建并返回全局词表字典 (词 -> ID)。
    
    参数:
    - sentences: 全部的句子文本列表
    
    返回:
    - word_map: 构建完的词表映射字典
    """
    word_freq = Counter()

    print("开始统计词频并进行中文分词...")
    # 对每句话使用 jieba 分词，并累加每个词出现的频率
    for sentence in tqdm(sentences):
        seg_list = jieba.cut(sentence)
        word_freq.update(list(seg_list))

    # 词表过滤：只保留出现频率大于 min_word_freq (定义在 config.py 中，默认为 3) 的词，以去除拼写错误和罕见噪声词。
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    
    # 构建 词 -> ID 映射词表。因为 0, 1, 2, 3 已经被特殊控制 Token 占用，普通词的 ID 索引从 4 开始。
    word_map = {k: v + 4 for v, k in enumerate(words)}
    
    # 插入特殊的控制字符
    word_map['<pad>'] = 0    # 填充对齐 Token
    word_map['<start>'] = 1  # 开始 Token (SOS)
    word_map['<end>'] = 2    # 结束 Token (EOS)
    word_map['<unk>'] = 3    # 未知词 Token (OOV)
    
    print(f"词表构建完成。词汇总数（含4个控制符）: {len(word_map)}")
    print(f"词表前10个保留词示例: {words[:10]}")

    return word_map


def build_samples(sentences, word_map):
    """
    将连续平铺的“问答交替”句子列表，构建并配对成用于训练的样本数据集（样本中包含 input 编码和 output 编码）。
    
    参数:
    - sentences: 问答平铺交替的句子列表。
      列表的偶数索引 (0, 2, 4...) 是问题的句子文本，
      列表的奇数索引 (1, 3, 5...) 是对应的回答的句子文本。
    - word_map: 词表映射字典
    
    返回:
    - samples: 包含数字编码问答对的列表。例如 [{'input': [...], 'output': [...]}]
    """
    print('开始构建并配对问答样本...')
    samples = []
    
    # 步长为 2 遍历列表，将 Question 和 Answer 配对
    for i in tqdm(range(0, len(sentences) - 1, 2)):
        sentence_in = sentences[i]
        seg_list = jieba.cut(sentence_in)
        tokens_in = encode_text(word_map, list(seg_list))  # 问句的 ID 序列
        
        sentence_out = sentences[i + 1]
        seg_list = jieba.cut(sentence_out)
        tokens_out = encode_text(word_map, list(seg_list)) # 答句的 ID 序列
        
        # 样本筛选策略：
        # 1. 问句长度和答句长度都必须小于等于 max_len（默认为 10）。过长的对话会让计算开销激增。
        # 2. 问句和答句中不能含有 <unk> 未知词 Token。过滤含未知词的样本可以提高模型生成的准确度和可读性。
        if len(tokens_in) <= max_len and len(tokens_out) <= max_len and UNK_token not in (tokens_in + tokens_out):
            samples.append({'input': list(tokens_in), 'output': list(tokens_out)})

    return samples


def build_sentences_xhj():
    """
    读取小黄鸡原始语料（.conv 文本格式），将其解析成平铺的 [问, 答, 问, 答...] 纯文本列表。
    """
    with open(xhj_corpus_loc, 'r', encoding="utf8") as f:
        sentences = f.readlines()
    print('小黄鸡原始文件行数: ' + str(len(sentences)))

    # 小黄鸡语料格式为：
    # E （代表一段对话开始）
    # M 问题文本
    # M 回答文本
    # 我们提取所有以 "M " 开头的行，并去除多余空白。
    sentences = [s[2:].strip() for s in sentences if len(s[1:].strip()) > 0]
    print('小黄鸡去空和控制字符后得到的对话行数: ' + str(len(sentences)))

    return sentences


def parse_ptt_lines(lines):
    """
    清洗并解析 PTT 中文对话语料文件中的每一行，提取出严格合法的问答对。
    
    参数:
    - lines: 原始文本行迭代对象
    
    返回:
    - 问答交替的平铺文本列表 ['Q1', 'A1', 'Q2', 'A2', ...]
    """
    result = []
    for s in lines:
        s = s.strip()
        if not s:
            continue
        # PTT 语料每行格式为：问题\t回答
        parts = s.split('\t')
        if len(parts) == 2:
            q = parts[0].strip()
            a = parts[1].strip()
            # 只有当问题和回答同时存在且非空时，才是合格样本
            if q and a:
                result.append(q)
                result.append(a)
    return result


def build_sentences_ptt():
    """
    读取并解析 PTT 问答语料，返回平铺的 [问, 答, 问, 答...] 文本列表。
    """
    with open(ptt_corpus_loc, 'r', encoding="utf8") as f:
        lines = f.readlines()
    print('PTT原始语料总行数: ' + str(len(lines)))

    result = parse_ptt_lines(lines)
    print(f'PTT解析清洗后得到的有效行数: {len(result)}')

    return result


def save_data(data, filename):
    """
    辅助函数：将数据保存为 JSON 格式文件。
    """
    with open(filename, 'w', encoding="utf8") as f:
        json.dump(data, f, indent=4)
    print('{} 数据已成功写入文件: {}.'.format(len(data), filename))


if __name__ == '__main__':
    # 接收命令行参数，支持命令行选择语料。如：uv run pre_process.py --corpus ptt
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', '-c', type=str, default='xhj', choices=['xhj', 'ptt'], help='选择语料: xhj (小黄鸡) 或 ptt (PTT语料)')
    args = parser.parse_args()

    # 1. 根据命令行选择加载不同语料
    if args.corpus == 'ptt':
        sentences = build_sentences_ptt()
    else:
        sentences = build_sentences_xhj()

    # 2. 统计并构建词表 (WORDMAP)
    word_map = build_wordmap(sentences)
    save_data(word_map, wordmap_loc)
    
    # 3. 将句子转换为 ID，并配对成样本对 (samples.json)
    samples = build_samples(sentences, word_map)
    save_data(samples, samples_loc)
