# encoding=utf-8
import torch
import torch.nn.functional as F
from torch import nn


class EncoderRNN(nn.Module):
    """
    编码器 (Encoder)：负责将输入的句子序列（数字ID）映射为包含上下文信息的稠密向量（Hidden States）。
    """
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers        # 门控循环单元 (GRU) 的堆叠层数
        self.hidden_size = hidden_size  # 隐藏层维度大小
        self.embedding = embedding      # 全局共享的嵌入层 (Embedding Layer)

        # 初始化 GRU（门控循环单元）。这里采用双向 GRU (bidirectional=True)。
        # 输入维度是 hidden_size，GRU 输出维度也是 hidden_size。
        # 相比单向 GRU，双向 GRU 能同时获取句子“从左到右”和“从右到左”的双向语境信息。
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        """
        前向传播函数。
        
        参数:
        - input_seq: 问句的 ID 序列张量。形状为 (seq_len, batch_size)
        - input_lengths: 句子的真实长度一维张量。形状为 (batch_size)
        - hidden: 初始隐藏状态向量。
        
        返回:
        - outputs: 双向融合后的所有时间步的编码器输出，形状为 (seq_len, batch_size, hidden_size)
        - hidden: GRU 的最终隐状态，形状为 (n_layers * 2, batch_size, hidden_size)
        """
        # 1. 词向量化：将词 ID 映射为对应的词嵌入向量。形状由 (seq_len, batch_size) 变为 (seq_len, batch_size, hidden_size)
        embedded = self.embedding(input_seq)
        
        # 2. 序列打包压缩：因为输入序列中有填充的 0 (PAD)，使用 pack_padded_sequence 告诉 GRU 忽略这些填充 0 的计算。
        # 这能大幅度节省计算力，并杜绝填充 0 的数据干扰模型隐藏状态的更新。
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        
        # 3. 运行双向 GRU。
        # outputs: 包含了每个时间步所有 GRU 节点的输出
        # hidden: 包含了每层最后一个时间步的 GRU 的隐状态（包含正反两向）
        outputs, hidden = self.gru(packed, hidden)
        
        # 4. 还原序列形状：将打包状态还原成一般的 Tensor。还原后 outputs 形状为 (seq_len, batch_size, hidden_size * 2)
        # 因为是双向，最后的维度是 hidden_size * 2 (正向 hidden_size + 反向 hidden_size)。
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        
        # 5. 双向输出求和：将 outputs 的正向部分与反向部分作相加求和（融合双向信息），使输出形状重新回到 (seq_len, batch_size, hidden_size)。
        # outputs[:, :, :self.hidden_size] 代表正向 GRU 的输出
        # outputs[:, :, self.hidden_size:] 代表反向 GRU 的输出
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        
        # 返回融合后的各个时间步输出，以及最终的隐状态
        return outputs, hidden


# Luong attention layer
class Attn(torch.nn.Module):
    """
    Luong 注意力层：用于计算 Decoder 当前隐状态与 Encoder 所有输出之间的关联权重。
    """
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "不是一种有效的注意力计算方式。")
        self.hidden_size = hidden_size
        
        # Luong Attention 提供了三种计算注意力相关度得分（Score）的经典公式：
        if self.method == 'general':
            # Score(h_t, h_s) = h_t^T * W * h_s
            # 初始化一个线性映射层 W
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            # Score(h_t, h_s) = v^T * tanh(W * [h_t; h_s])
            # 初始化拼接后的投影矩阵 W，以及一个可学习的参数向量 v
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        """
        'dot' 点积机制计算法：直接将 Decoder 隐状态与 Encoder 各时间步输出作点积乘法。
        """
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        """
        'general' 机制计算法：在作乘法之前引入一个可学习的 Linear 映射矩阵进行变换。
        """
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        """
        'concat' 拼接机制计算法：把 Decoder 隐状态与 Encoder 输出拼接，再经过一个线性层和 tanh 激活，最后与参数向量 v 点乘。
        """
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        """
        前向传播计算相关度，并通过 Softmax 归一化为概率权重值。
        
        参数:
        - hidden: Decoder 当前的 GRU 输出隐状态 (1, batch_size, hidden_size)
        - encoder_outputs: Encoder 的所有时间步隐藏状态 (seq_len, batch_size, hidden_size)
        
        返回:
        - 注意力分配权重分配值形状为 (batch_size, 1, seq_len)
        """
        # 1. 根据配置的方法计算相似度能量分 (Energy score)
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # 2. 转置张量维度，从 (seq_len, batch_size) 变成 (batch_size, seq_len)
        attn_energies = attn_energies.t()

        # 3. 对 seq_len 维度进行 Softmax。将分数转化为概率分布（权重之和为1.0），并在中间增加一维输出。
        # 输出的形状为 (batch_size, 1, seq_len)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttnDecoderRNN(nn.Module):
    """
    解码器 (Decoder)：负责把 Encoder 的特征表示逐步还原为目标文本序列（回答句子）。
    """
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # 保存各个超参数配置
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # 1. 声明网络层
        self.embedding = embedding              # 与 Encoder 共享相同的 Embedding 层
        self.embedding_dropout = nn.Dropout(dropout) # 对嵌入向量随机失活
        
        # Decoder 采用单向 GRU，因为在预测时不需要也不能知道未来的后文信息。
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        
        # 拼接层：用于将注意力“上下文向量”与当前的“GRU隐状态”拼接，再映射回 hidden_size 维度
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        
        # 输出投影层：将隐向量投射到词表大小维度的空间 (output_size)，其每个维度的值表示词表中对应词的预测得分。
        self.out = nn.Linear(hidden_size, output_size)

        # 2. 实例化注意力计算层
        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        """
        解码前向传播。注意：在解码或推理时，我们是一个时间步（即一个词）一个时间步地循环运行 Decoder 的。
        
        参数:
        - input_step: 当前步骤输入的词 ID 张量。形状为 (1, batch_size)
        - last_hidden: 上一步 Decoder 留下的隐状态。形状为 (n_layers, batch_size, hidden_size)
        - encoder_outputs: 编码器输出的所有状态。用于计算注意力。 (seq_len, batch_size, hidden_size)
        
        返回:
        - output: 本时间步对词表中每个词语的预测概率分布，形状为 (batch_size, output_size)
        - hidden: 本时间步产生的最新隐状态，用于传递给下一步解码。 (n_layers, batch_size, hidden_size)
        """
        # 1. 获取输入词的 Embeddings，并传入 Dropout 层。形状为 (1, batch_size, hidden_size)
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        
        # 2. 将当前词向量与上一步的隐状态送入单向 GRU 计算，得到新状态 rnn_output 和 hidden
        # rnn_output 形状为 (1, batch_size, hidden_size)
        # hidden 形状为 (n_layers, batch_size, hidden_size)
        rnn_output, hidden = self.gru(embedded, last_hidden)
        
        # 3. 核心注意力计算：计算当前步的 rnn_output 对整个 encoder_outputs 序列的关注度权重
        # attn_weights 形状为 (batch_size, 1, seq_len)
        attn_weights = self.attn(rnn_output, encoder_outputs)
        
        # 4. 加权求和：利用注意力权重对 Encoder 隐藏状态进行矩阵乘法（bmm），获取当前的上下文向量 (Context Vector)
        # 相当于对编码器输出进行局部加权聚合，生成本步骤需要重点关注的特征。
        # context 形状为 (batch_size, 1, hidden_size)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        
        # 5. 拼接与降维：将 rnn_output 和 context 进行拼接，输入线性层映射，以融合注意力特征与隐状态。
        rnn_output = rnn_output.squeeze(0)  # 去除第0维，形状变成 (batch_size, hidden_size)
        context = context.squeeze(1)     # 去除第1维，形状变成 (batch_size, hidden_size)
        
        # 拼接后维度变成 hidden_size * 2，形状为 (batch_size, hidden_size * 2)
        concat_input = torch.cat((rnn_output, context), 1)
        # 用 tanh 激活激活并降维为 (batch_size, hidden_size)
        concat_output = torch.tanh(self.concat(concat_input))
        
        # 6. 计算输出：将降维特征通过输出 Linear 层，获得各词预测分，并用 Softmax 转化成总和为 1.0 的概率分布
        output = self.out(concat_output)
        output = F.softmax(output, dim=1) # 最终大小为 (batch_size, output_size)
        
        # 返回词分布概率，以及最新的隐状态供下一步迭代使用
        return output, hidden
