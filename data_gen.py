# encoding=utf-8
import itertools
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="jieba")

import jieba
from torch.utils.data import Dataset
from config import *


def indexesFromSentence(voc, sentence):
    """
    将普通句子分词，并使用 Voc 词表对象转换成对应的 ID 列表，末尾加上 EOS 结束标志。
    
    参数:
    - voc: Voc 词表管理对象
    - sentence: 句子文本
    
    返回:
    - 词 ID 列表，末尾追加 EOS_token (ID 2)
    """
    return [voc.word2index[word] for word in jieba.cut(sentence)] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    """
    对多条长度不一致的句子的 ID 列表进行对齐填充（Padding）。
    我们使用 zip_longest 把它转置并填充。
    
    输入:
    - l: 一个二维列表，外层是 Batch，内层是每句话的词 ID 列表。例如: [[Q1_w1, Q1_w2], [Q2_w1, Q2_w2, Q2_w3]]
    
    返回:
    - 一个转置并填充后的二维列表。转置是为了让形状变成 (max_sequence_len, batch_size)，这在 PyTorch 的 RNN 运算中更高效。
      缺失的位置会被填充为 PAD_token (ID 0)。
    """
    # itertools.zip_longest 会提取每个列表的第 i 个元素组成新的 tuple，如果某些列表长度不够，则使用 fillvalue 填充。
    # 这相当于对列表矩阵做了一个转置，并在短序列尾部填充 0。
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l):
    """
    为填充好的二维 ID 列表矩阵生成一个等大的二值掩码（Mask）矩阵。
    有实际内容（不是 PAD_token）的位置为 1，被填充（是 PAD_token）的位置为 0。
    
    参数:
    - l: 对齐填充转置后的二维列表 (max_sequence_len, batch_size)
    
    返回:
    - 对应位置的二值化列表矩阵 (0/1)
    """
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)  # PAD_token 处记为 0
            else:
                m[i].append(1)  # 真实词处记为 1
    return m


# Returns padded input sequence tensor and lengths
def inputVar(indexes_batch):
    """
    处理输入问句（Question）的 Batch：进行对齐填充，并获取每个句子的真实长度（去填充前）。
    
    参数:
    - indexes_batch: 一批句子的 ID 列表 [[ID, ID...], [ID, ID...]]
    
    返回:
    - padVar: 填充并转置后的张量 (Tensor)，形状为 (max_seq_len, batch_size)
    - lengths: 一个一维张量，记录该 Batch 内每个句子的真实有效长度（不含填充的 0），
               用于后续的 pack_padded_sequence 压缩计算。
    """
    # 1. 提取每个句子的实际长度，生成长度 Tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # 2. 填充数据并转置
    padList = zeroPadding(indexes_batch)
    # 3. 转化为 PyTorch 能够运算的长整型 Tensor
    padVar = torch.LongTensor(padList)
    return padVar, lengths


# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(indexes_batch):
    """
    处理输出答句（Answer）的 Batch：进行对齐填充，并创建对应的 Loss 计算掩码（Mask）。
    
    参数:
    - indexes_batch: 一批句子的 ID 列表
    
    返回:
    - padVar: 填充转置后的张量，形状为 (max_target_len, batch_size)
    - mask: 二值掩码 Tensor，用于告诉 Loss 函数在计算损失时忽略填充的无效 0 位置。形状为 (max_target_len, batch_size)。
    - max_target_len: 本 Batch 中答句的最大长度。
    """
    # 1. 获取本 Batch 答句的最长长度
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    # 2. 填充并转置数据
    padList = zeroPadding(indexes_batch)
    # 3. 得到二值化的 Mask 列表
    mask = binaryMatrix(padList)
    # 4. 转换为字节型掩码 Tensor (ByteTensor)
    mask = torch.ByteTensor(mask)
    # 5. 转换为长整型数据 Tensor
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


# Returns all items for a given batch of pairs
def batch2TrainData(pair_batch):
    """
    将一整批 (Question, Answer) 元组数据打包处理，转换成 PyTorch 模型能够直接训练的张量形式。
    
    参数:
    - pair_batch: 包含若干个 (question_tokens, answer_tokens) 的列表。
    
    返回:
    - inp: 处理并填充后的输入问句张量 (max_len_in, batch_size)
    - lengths: 输入问句的实际长度一维张量 (batch_size)
    - output: 处理并填充后的输出答句张量 (max_len_out, batch_size)
    - mask: 用于忽略答句填充部分 Loss 计算的掩码张量 (max_len_out, batch_size)
    - max_target_len: 答句的最大长度（即解码的最大循环次数）
    """
    # 关键机制：PyTorch 的 pack_padded_sequence 要求在进行压缩前，必须将输入 Batch 按照句子实际有效长度从大到小排序。
    # 排序键 key 使用问句（x[0]）的长度，设置 reverse=True 表示降序排列。
    pair_batch.sort(key=lambda x: len(x[0]), reverse=True)
    
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])  # 问句
        output_batch.append(pair[1])  # 答句
        
    # 处理输入问句（生成数据和长度信息）
    inp, lengths = inputVar(input_batch)
    # 处理输出答句（生成数据、掩码、和最大长度）
    output, mask, max_target_len = outputVar(output_batch)
    
    return inp, lengths, output, mask, max_target_len


class ChatbotDataset(Dataset):
    """
    PyTorch 数据集类。负责将大 JSON 对话样本文件加载进来，并将其切分为训练集和验证集。
    """
    def __init__(self, split):
        self.split = split
        assert self.split in {'train', 'valid'}

        print('正在从本地加载 {} 样本: {} ...'.format(self.split, samples_loc))
        self.samples = json.load(open(samples_loc, 'r'))

        # 动态切分：将数据集以 95% 作为训练集，余下的 5% 作为评估/验证集
        total_samples = len(self.samples)
        num_train = int(total_samples * 0.95)

        if split == 'train':
            self.samples = self.samples[:num_train]
        else:
            self.samples = self.samples[num_train:]

    def __getitem__(self, idx):
        """
        根据索引 idx 提取单个样本的 (input, output) 元组。
        由 DataLoader 批量加载并使用 collate_fn 整理。
        """
        sample = self.samples[idx]
        return sample['input'], sample['output']

    def __len__(self):
        """
        计算样本总数。
        """
        return len(self.samples)


if __name__ == '__main__':
    # 示例运行：实例化 ChatbotDataset，并提取第一个批次，打印各项参数的维度
    print('loading {} samples'.format('valid'))
    samples = json.load(open(samples_loc, 'r'))
    pair_batch = []
    for i in range(5):
        sample = samples[i]
        pair_batch.append((sample['input'], sample['output']))

    # 实例演示
    small_batch_size = 5
    batches = batch2TrainData(pair_batch)
    input_variable, lengths, target_variable, mask, max_target_len = batches

    print("--- 批处理张量演示 ---")
    print("输入问句张量（填充转置后）:\n", input_variable)
    print("问句真实长度（不含0）:", lengths)
    print("目标答句张量（填充转置后）:\n", target_variable)
    print("Loss掩码张量（1代表有效词，0代表填充0）:\n", mask)
    print("当前批次答句最大长度 (max_target_len):", max_target_len)
