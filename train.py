# encoding=utf-8
from torch import optim
from data_gen import ChatbotDataset
from models import EncoderRNN, LuongAttnDecoderRNN
from utils import *


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, encoder_optimizer,
          decoder_optimizer):
    """
    训练单个批次 (Batch) 数据的核心逻辑。
    
    参数:
    - input_variable: 问句 Tensor 矩阵，形状为 (seq_len_in, batch_size)
    - lengths: 问句的实际长度 Tensor
    - target_variable: 答句 Tensor 矩阵，形状为 (seq_len_out, batch_size)
    - mask: 答句掩码矩阵，用于过滤 PAD 字符的损失。形状同上
    - max_target_len: 答句中最长句子的长度值（即最大迭代预测步数）
    - encoder, decoder: 编码器和解码器神经网络模型
    - encoder_optimizer, decoder_optimizer: 编码器与解码器的优化器 (Adam)
    
    返回:
    - 本批次的平均每词交叉熵损失值 (Average loss per token)
    """
    # ----------------------------------------------------
    # 步骤 1. 梯度归零
    # ----------------------------------------------------
    # PyTorch 默认会累加历史梯度。因此在处理每一个新 Batch 时，必须先清零梯度，否则模型会在错误的梯度方向上更新。
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # ----------------------------------------------------
    # 步骤 2. 数据拷贝至运行设备 (GPU 或 CPU)
    # ----------------------------------------------------
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # 初始化记录指标的局部变量
    loss = 0
    print_losses = []
    n_totals = 0

    # ----------------------------------------------------
    # 步骤 3. 前向传播：编码问句 (Encoder)
    # ----------------------------------------------------
    # 输入问句和实际长度，获取问句各步输出隐状态 outputs，以及最顶层最后一个时间步隐状态 hidden。
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # ----------------------------------------------------
    # 步骤 4. 初始化解码器输入与初隐状态 (Decoder Init)
    # ----------------------------------------------------
    # 解码是一个循环预测的过程（一个字接一个字输出）。
    # 第一步：我们的初始输入应是 SOS_token（开始符号）。构造一个形状为 (1, batch_size) 的张量。
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # 初始隐状态：我们把 Encoder 的最终隐藏状态（经过层数截断）传递给 Decoder，做为 Decoder 的初始“记忆”。
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # ----------------------------------------------------
    # 步骤 5. 教师迫导 (Teacher Forcing) 策略决策
    # ----------------------------------------------------
    # 在本批次迭代中，判断是否使用 Teacher Forcing（根据 teacher_forcing_ratio 随机抽样）。
    # 开启时：直接喂给模型真实答案词作为下一步的输入；
    # 关闭时：使用模型在上一步自己预测最像的那个词作为下一步的输入。
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # ----------------------------------------------------
    # 步骤 6. 循环解码计算 Loss
    # ----------------------------------------------------
    if use_teacher_forcing:
        # 【Teacher Forcing 模式】
        # 在这中模式下，下一步的 decoder_input 被直接强行设为目标真实的词。
        for t in range(max_target_len):
            # 前向计算：传入当前词，上一步隐状态和 encoder 记忆，获取当前字预测概率和最新隐状态
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # 教师迫导：强行把当前时间步的目标真实词 target_variable[t] 作为下一次循环的输入词。
            # .view(1, -1) 将其维度调整为符合要求的 (1, batch_size)
            decoder_input = target_variable[t].view(1, -1)
            
            # 掩码损失计算：利用二值化 Mask 掩码矩阵过滤无意义的 PAD_token (0) 损失，只累计有效预测损失。
            # maskNLLLoss 在 utils.py 中定义。
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        # 【常规自回归解码模式】
        # 将模型自己预测出的最佳词作为下一步的输入。
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # 通过 topk(1) 获取当前概率分布中最大（即概率最高）的词 ID 及其索引索引
            _, topi = decoder_output.topk(1)
            
            # 将这个预测出来的词 ID 提取并重构为下一次循环的输入。形状依然为 (1, batch_size)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            
            # 掩码损失计算
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # ----------------------------------------------------
    # 步骤 7. 反向传播与梯度更新
    # ----------------------------------------------------
    # 反向传播计算所有可训练权重的梯度：
    loss.backward()

    # 梯度剪裁：如果梯度过大，会导致数值溢出崩溃（NaN）。
    # clip_grad_norm_ 将梯度范数强行缩减到 clip 值（50.0），确保神经网络更新时的安全稳定。
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # 优化器工作：利用计算好的梯度来调整权重参数。
    encoder_optimizer.step()
    decoder_optimizer.step()

    # 返回该批次中每个非填充字符的平均 Loss 损失值
    return sum(print_losses) / n_totals


def validate(val_loader, encoder, decoder):
    """
    在验证集 (Validation Set) 上计算损失。
    验证过程中，只执行前向传播，不计算梯度，也不更新参数。
    """
    # 开启 torch.no_grad() 临时关闭梯度追踪引擎。这能显著节省显存和计算时间。
    with torch.no_grad():
        # 设为评估状态。这会让 Dropout 机制自动失效，使行为确定化。
        encoder.eval()
        decoder.eval()

        batch_time = AverageMeter()
        losses = AverageMeter()
        print_losses = []
        n_totals = 0

        start = time.time()

        # 遍历验证集中的所有批次 (Batches)
        for i_batch in range(val_loader.__len__()):
            input_variable, lengths, target_variable, mask, max_target_len = val_loader.__getitem__(i_batch)
            input_variable = input_variable.to(device)
            lengths = lengths.to(device)
            target_variable = target_variable.to(device)
            mask = mask.to(device)

            loss = 0

            # 前向计算问句
            encoder_outputs, encoder_hidden = encoder(input_variable, lengths)
            
            # 问答序列初始化
            decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            decoder_hidden = encoder_hidden[:decoder.n_layers]

            # 验证集上绝不能使用 Teacher Forcing！必须全用模型自己的真实预测来作为下一步的输入。
            for t in range(max_target_len):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
                decoder_input = decoder_input.to(device)
                
                mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal

            loss = sum(print_losses) / n_totals
            # 累加指标
            losses.update(loss, max_target_len)
            batch_time.update(time.time() - start)

            start = time.time()

    # 返回验证集平均损失
    return sum(print_losses) / n_totals


def evaluate(searcher, sentence, max_length=max_len):
    """
    单步预测推理函数。用于测试或者在交互界面中回答用户的提问。
    """
    # 格式化输入的文本句子，切词并转为 ID Batch (形状为 1xN)
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # 获取问句长度
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # 转置形状以适配模型的 expectation 
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    
    # 运行 GreedySearchDecoder (贪心搜索解码器) 预测出最佳 ID 序列和相似度分数
    tokens, scores = searcher(input_batch, lengths, max_length)
    
    # 将输出的 ID 数字列表解码成具体的文字词汇，跳过填充控制符 <pad> 和开始控制符 <start>
    decoded_words = [voc.index2word[token.item()] for token in tokens
                     if token != EOS_token and token != PAD_token]
    return decoded_words


def main():
    # 1. 实例化 PyTorch 的 Dataset 和加载器
    train_loader = ChatbotDataset('train')
    val_loader = ChatbotDataset('valid')

    # 2. 初始化嵌入矩阵权重（Embedding）。所有的词在被送进 RNN 之前都将在这个表中转换成稠密向量。
    embedding = nn.Embedding(voc.num_words, hidden_size)

    # 3. 初始化 Encoder 和 Decoder 神经网络模型
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

    # 将网络参数复制到指定计算设备中
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # 4. 初始化 Adam 优化器 (Optimizers)。分别控制两个模块的训练更新步伐。
    print('正在创建 Adam 优化器...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

    # 加载 Checkpoint 以便断点续训
    epoch_start = start_epoch
    if loadFilename:
        if os.path.exists(loadFilename):
            print('正在从本地加载 Checkpoint: {} ...'.format(loadFilename))
            checkpoint = torch.load(loadFilename, map_location=device)
            epoch_start = checkpoint['epoch'] + 1
            encoder.load_state_dict(checkpoint['en'])
            decoder.load_state_dict(checkpoint['de'])
            encoder_optimizer.load_state_dict(checkpoint['en_opt'])
            decoder_optimizer.load_state_dict(checkpoint['de_opt'])
            if 'voc' in checkpoint:
                voc.__dict__.update(checkpoint['voc'])
            print('Checkpoint 加载成功，将从 Epoch {} 继续训练。'.format(epoch_start))
        else:
            print('警告: 指定的 Checkpoint 文件未找到，将从头开始训练: {}'.format(loadFilename))

    # 初始化计时和计量指标记录工具
    print('数据初始化就绪...')
    batch_time = AverageMeter()  
    losses = AverageMeter()  

    # 5. 开启 Epoch 大循环
    for epoch in range(epoch_start, epochs):
        # 激活训练状态（允许激活 Dropout 等非确定层动作）
        encoder.train()
        decoder.train()

        start = time.time()

        # 遍历训练集 Batch 列表
        for i in range(train_loader.__len__()):
            input_variable, lengths, target_variable, mask, max_target_len = train_loader.__getitem__(i)
            # 执行单步批训练，获得当前损失值
            loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder,
                         encoder_optimizer, decoder_optimizer)

            # 记录本步指标
            losses.update(loss, max_target_len)
            batch_time.update(time.time() - start)

            start = time.time()

            # 按配置规律打印日志
            if i % print_every == 0:
                print('[{0}] Epoch: [{1}][{2}/{3}]\t'
                      '批处理耗时 {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      '交叉熵 Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(timestamp(), epoch, i, len(train_loader),
                                                                       batch_time=batch_time,
                                                                       loss=losses))
        # ----------------------------------------------------
        # 6. 每个 Epoch 结束后，在验证集上计算一次 Loss 损失
        # ----------------------------------------------------
        val_loss = validate(val_loader, encoder, decoder)
        print('\n * 验证集损失 - {loss:.3f}\n'.format(loss=val_loss))

        # ----------------------------------------------------
        # 7. 自动展示：随机抽取10句验证集句子让模型作答，直观评估当前模型水平
        # ----------------------------------------------------
        searcher = GreedySearchDecoder(encoder, decoder)
        for sentence in pick_n_valid_sentences(10):
            decoded_words = evaluate(searcher, sentence)
            print('人类发问: {}'.format(sentence))
            print('机器人作答: {}'.format(''.join(decoded_words)))

        # ----------------------------------------------------
        # 8. 保存模型权重 Checkpoint
        # ----------------------------------------------------
        if epoch % save_every == 0:
            directory = save_dir
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'epoch': epoch,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc': voc.__dict__
            }, os.path.join(directory, '{}_{}_{}.tar'.format('checkpoint', epoch, val_loss)))


if __name__ == '__main__':
    # 接收并解析命令行参数。例如：uv run train.py --corpus ptt
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', '-c', type=str, default='xhj', choices=['xhj', 'ptt'], help='选择训练的语料: xhj (小黄鸡) 或 ptt (PTT语料)')
    args = parser.parse_args()
    main()
