# encoding=utf-8
import os
import random
import time
import torch
from torch import optim
from torch.utils.data import DataLoader
from data_gen import ChatbotDataset, batch2TrainData
from models import EncoderRNN, LuongAttnDecoderRNN
from utils import *

def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, encoder_optimizer,
          decoder_optimizer, scaler=None):
    """
    训练单个批次 (Batch) 数据的核心逻辑。包含前向传播、损失计算、反向传播与优化器参数更新。
    
    参数详细解释 (适合新手学习):
    - input_variable: 问句的 Word ID 矩阵，类型为 LongTensor，形状为 (seq_len_in, batch_size)。
                      每一列代表一个句子，每一行对应一个时间步的单词。
    - lengths: 每一个问句的实际有效长度（不计入 PAD 填充符）的一维 Tensor，形状为 (batch_size)。
               用于 PackPaddedSequence 压缩 RNN 计算以省去填充符的运算。
    - target_variable: 答句的 Word ID 矩阵，类型为 LongTensor，形状为 (seq_len_out, batch_size)。
    - mask: 答句的掩码二值矩阵，形状同 target_variable，为 (seq_len_out, batch_size)。
            0 代表该位置是填充占位符 <pad>，1 代表是真实有效的单词。
    - max_target_len: 当前 Batch 答句中最长的句子长度值，决定了解码器最大需要展开循环预测多少步。
    - encoder, decoder: 编码器与解码器神经网络模型对象。
    - encoder_optimizer, decoder_optimizer: 控制编码器与解码器网络参数梯度更新步伐的优化器 (Adam)。
    - scaler: 自动混合精度 (AMP) 梯度缩放器。若不为 None，将以半精度 (FP16) 计算以提速。
    
    返回:
    - 本批次每个有效预测词的平均交叉熵损失值 (Average loss per token)。
    """
    # ----------------------------------------------------
    # 步骤 1. 清空历史梯度 (Gradient Zeroing)
    # ----------------------------------------------------
    # PyTorch 在反向传播时默认会累加每次计算得到的梯度。
    # 因此在处理每一个新 Batch 数据前，必须主动将上一轮的梯度清零，防止参数朝错误的方向更新。
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # ----------------------------------------------------
    # 步骤 2. 数据拷贝至目标运行设备 (GPU 或 CPU)
    # ----------------------------------------------------
    # config.py 中配置的 device 会根据当前硬件自动选择 cuda (显卡) 或 cpu (处理器)。
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # 初始化记录指标的局部变量
    loss = 0            # 累加每个时间步的损失 Tensor
    print_losses = []   # 保存每个时间步的损失实数，用于最终日志打印
    n_totals = 0        # 累计当前 batch 中包含的实际有效词汇（非 PAD）的总个数

    # ----------------------------------------------------
    # 步骤 3. 混合精度上下文包围 (AMP Autocast)
    # ----------------------------------------------------
    # 在 with autocast(): 作用域下，PyTorch 会自动将矩阵乘法、线性层等运算的精度从 32位浮点 (FP32)
    # 降低到 16位浮点 (FP16)。这在 GPU 上能显著释放 Tensor Cores 的硬件加速能力，极大提速并腰斩显存开销。
    with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
        
        # ----------------------------------------------------
        # 步骤 4. 前向传播：编码问句 (Encoder Forward)
        # ----------------------------------------------------
        # 传入问句矩阵和长度，得到：
        # - encoder_outputs: 每一个时间步的输出表示，形状为 (seq_len_in, batch_size, hidden_size)
        # - encoder_hidden: 最后一层最终的隐藏状态，包含问句的全局语义向量表示
        encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

        # ----------------------------------------------------
        # 步骤 5. 初始化解码器输入与初始隐状态 (Decoder Init)
        # ----------------------------------------------------
        # 解码是一个自回归序列预测任务（逐字输出，上一步输出作为下一步输入）。
        # 第一个时间步：强行输入 SOS_token（句子起始控制符）。构造形状为 (1, batch_size) 的张量。
        decoder_input = torch.LongTensor([[SOS_token for _ in range(input_variable.size(1))]])
        decoder_input = decoder_input.to(device)
        
        # 解码器的初始隐藏状态直接复制自 Encoder 的最终隐藏状态（并截取所需的 decoder 层数大小）
        decoder_hidden = encoder_hidden[:decoder.n_layers]

        # ----------------------------------------------------
        # 步骤 6. 导师迫导 (Teacher Forcing) 策略决策
        # ----------------------------------------------------
        # 在训练早期阶段，模型预测通常很不准。如果下一步的输入总是使用模型上一步的错误预测，
        # 会导致“一步错步步错”的误差漂移现象，使得网络极难收敛。
        # 解决方法：按一定概率 (teacher_forcing_ratio) 开启 Teacher Forcing，
        # 在计算下一步输入时，直接“强行喂给”它真实的标准答案，从而拉回正轨。
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        # ----------------------------------------------------
        # 步骤 7. 循环解码计算 Loss
        # ----------------------------------------------------
        if use_teacher_forcing:
            # 【启用 Teacher Forcing 模式】
            for t in range(max_target_len):
                # 传入当前单词输入、上一时刻隐藏状态和编码器的全局表示（Attention 机制）
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # 导师迫导：下一次循环的输入直接设为标准答案中对应时间步单词 target_variable[t]
                # .view(1, -1) 将其调整为 (1, batch_size) 的输入形状
                decoder_input = target_variable[t].view(1, -1)
                
                # 掩码损失计算：利用 Mask 排除 PAD 字符的干扰，只对有效字符累计 Loss
                mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal
        else:
            # 【常规自回归解码模式】
            for t in range(max_target_len):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # 选择概率最高（最可能）的词汇 ID 索引值作为预测值
                _, topi = decoder_output.topk(1)
                
                # 下一次循环的输入设置为模型当前时间步的预测词汇 (直接在 GPU 上进行维度变换，规避 CPU 循环与数据复制)
                decoder_input = topi.view(1, -1)
                
                # 掩码损失计算
                mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal

    # ----------------------------------------------------
    # 步骤 8. 反向传播与梯度更新 (支持 AMP 缩放)
    # ----------------------------------------------------
    if scaler is not None:
        # 当使用 FP16 半精度训练时，某些参数梯度可能非常微小，这会导致数值下溢（变为 0 导致权重无法更新）。
        # 解决方法：GradScaler 会在反向传播前将损失值乘以一个比例因子进行放大 (scaler.scale(loss))。
        scaler.scale(loss).backward()
        
        # 在用优化器步进前，需要将梯度先 unscale 缩放回原始大小，以进行正确的梯度裁剪操作。
        scaler.unscale_(encoder_optimizer)
        scaler.unscale_(decoder_optimizer)
        
        # 梯度裁剪 (Gradient Clipping)：如果序列很长，反向传播产生的梯度可能极高，导致参数爆炸 (NaN)。
        # clip_grad_norm_ 将梯度向量的范数裁剪到指定的安全限制范围 (clip=50.0)，防止数值爆炸。
        _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
        
        # 优化器参数更新：使用调整后的真实梯度更新模型参数，并更新缩放器的内部参数
        scaler.step(encoder_optimizer)
        scaler.step(decoder_optimizer)
        scaler.update()
    else:
        # 常规 FP32 精度的反向传播与步进逻辑（主要在 CPU 或未启用 CUDA 时运行）
        loss.backward()
        _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
        encoder_optimizer.step()
        decoder_optimizer.step()

    # 返回这一批次内所有有效预测字符的平均交叉熵损失值
    return sum(print_losses) / n_totals


def validate(val_loader, encoder, decoder):
    """
    在验证集 (Validation Set) 上计算损失。
    验证过程中，只执行前向传播计算损失，不进行反向传播，不更新任何网络权重参数。
    """
    # with torch.no_grad() 告诉 PyTorch 临时关闭它的梯度追踪引擎。
    # 这能极大减少推理计算时的内存开销，并显著提速。
    with torch.no_grad():
        # 设为评估状态。这会让 Dropout、Batch Normalization 等只在训练状态生效的结构自动失效。
        encoder.eval()
        decoder.eval()

        batch_time = AverageMeter()
        losses = AverageMeter()
        print_losses = []
        n_totals = 0

        start = time.time()

        # 遍历验证集中的所有批次 (Batches)
        for batch in val_loader:
            input_variable, lengths, target_variable, mask, max_target_len = batch
            input_variable = input_variable.to(device)
            lengths = lengths.to(device)
            target_variable = target_variable.to(device)
            mask = mask.to(device)

            loss = 0

            # 验证集推理同样启用 autocast 进行加速
            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                encoder_outputs, encoder_hidden = encoder(input_variable, lengths)
                
                decoder_input = torch.LongTensor([[SOS_token for _ in range(input_variable.size(1))]])
                decoder_input = decoder_input.to(device)
                decoder_hidden = encoder_hidden[:decoder.n_layers]

                # 注意：验证集上绝不能使用 Teacher Forcing！
                # 必须完全利用模型自己的真实预测来作为下一步的输入，以保证评估结果的真实与中立。
                for t in range(max_target_len):
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden, encoder_outputs
                    )
                    _, topi = decoder_output.topk(1)
                    # 直接在 GPU 上进行维度变换，规避 CPU 循环与数据复制
                    decoder_input = topi.view(1, -1)
                    
                    mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
                    loss += mask_loss
                    print_losses.append(mask_loss.item() * nTotal)
                    n_totals += nTotal

            loss = sum(print_losses) / n_totals
            losses.update(loss, max_target_len)
            batch_time.update(time.time() - start)
            start = time.time()

    return sum(print_losses) / n_totals


def evaluate(searcher, sentence, max_length=max_len):
    """
    单步预测推理函数。用于对人机交互界面中用户输入的单句话进行自回归预测并作答。
    """
    # 1. 格式化输入的文本，分词并转为 ID 列表 (例如: [12, 45, 9])
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # 2. 问句长度一维张量
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # 3. 转置形状，以匹配模型的输入期望维度：(seq_len_in, batch_size=1)
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    
    # 4. 调用 GreedySearchDecoder 推理计算，获得预测 ID 序列和打分
    tokens, scores = searcher(input_batch, lengths, max_length)
    
    # 5. 将输出的 ID 列表翻译解码回对应的文字，跳过填充符与句子开始标记
    decoded_words = [voc.index2word[token.item()] for token in tokens
                     if token != EOS_token and token != PAD_token]
    return decoded_words


def main():
    # ----------------------------------------------------
    # 1. 实例化 PyTorch 的 Dataset 和加载器
    # ----------------------------------------------------
    # 使用 PyTorch 异步多进程 DataLoader 取代手动提取。
    # - num_workers: 启动的 CPU 数据预处理后台线程数，实现后台提取与 GPU 前台计算的并行化。
    # - pin_memory=True: 将 CPU 加载的数据块锁页在物理内存中。当拷贝数据到 GPU 时，
    #   显卡可以通过 DMA (直接内存访问) 机制以极高带宽从锁页内存直接传输，极大降低 CPU-GPU 数据拷贝时延。
    train_loader = DataLoader(
        ChatbotDataset('train'), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=workers, 
        collate_fn=batch2TrainData, 
        pin_memory=True,
        persistent_workers=(workers > 0)
    )
    val_loader = DataLoader(
        ChatbotDataset('valid'), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=workers, 
        collate_fn=batch2TrainData, 
        pin_memory=True,
        persistent_workers=(workers > 0)
    )

    # ----------------------------------------------------
    # 2. 初始化嵌入矩阵权重 (Embedding Layer)
    # ----------------------------------------------------
    # 将词 ID (0 ~ 词表大小) 转换为固定维度 (hidden_size) 的稠密实数稠密向量。
    # 这是神经网络处理离散文本符号的基石。
    embedding = nn.Embedding(voc.num_words, hidden_size)

    # ----------------------------------------------------
    # 3. 初始化 Encoder 和 Decoder 神经网络模型
    # ----------------------------------------------------
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

    # 将网络参数拷贝到指定的设备 (CPU/GPU) 内存中
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # ----------------------------------------------------
    # 4. 初始化 Adam 优化器 (Optimizers)
    # ----------------------------------------------------
    print('正在创建 Adam 优化器...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

    # ----------------------------------------------------
    # 5. 加载 Checkpoint 以便断点续训 (自动检测与恢复)
    # ----------------------------------------------------
    epoch_start = start_epoch
    if loadFilename:
        resolved_path = loadFilename
        is_interrupted_cp = False

        # 如果配置为 'best' 或 'latest'，优先自动定位 models/ 目录下的最优权重或最新中断备份
        interrupted_path = os.path.join(save_dir, 'checkpoint_interrupted.tar')
        if loadFilename in ('best', 'latest') and os.path.exists(interrupted_path):
            resolved_path = interrupted_path
            is_interrupted_cp = True
            print('检测到上次意外中断的恢复包，将优先进行自动恢复：{}'.format(resolved_path))
        elif loadFilename in ('best', 'latest'):
            resolved_path = find_checkpoint(save_dir, mode=loadFilename)
            if resolved_path:
                print('自动检测到 {} Checkpoint: {}'.format(loadFilename, resolved_path))
            else:
                print('未在 {} 目录下找到任何有效的 Checkpoint，将从头开始训练。'.format(save_dir))

        if resolved_path and os.path.exists(resolved_path):
            print('正在从本地加载 Checkpoint: {} ...'.format(resolved_path))
            checkpoint = torch.load(resolved_path, map_location=device)
            if is_interrupted_cp:
                # 如果是中断恢复包，说明当前 Epoch 还没有跑完就被中断了。
                # 重新启动时，应该从当前 Epoch 继续重新跑，而不是直接跳过该 Epoch 到下一轮。
                epoch_start = checkpoint['epoch']
            else:
                # 如果是正常的、已经完整跑完的 Epoch 存档，则从下一个 Epoch 开始训练。
                epoch_start = checkpoint['epoch'] + 1
            encoder.load_state_dict(checkpoint['en'])
            decoder.load_state_dict(checkpoint['de'])
            encoder_optimizer.load_state_dict(checkpoint['en_opt'])
            decoder_optimizer.load_state_dict(checkpoint['de_opt'])
            if 'voc' in checkpoint:
                voc.__dict__.update(checkpoint['voc'])
            print('Checkpoint 加载成功，将从 Epoch {} 继续训练。'.format(epoch_start))

            # 成功加载临时中断包后，主动清理它，以防未来开启全新训练时被误读
            if is_interrupted_cp:
                try:
                    os.remove(resolved_path)
                    print('已成功清理临时中断恢复包。')
                except OSError as e:
                    print('清理临时中断恢复包失败: {}'.format(e))
        elif resolved_path:
            print('警告: 指定的 Checkpoint 文件未找到，将从头开始训练: {}'.format(resolved_path))

    # ----------------------------------------------------
    # 6. 使用 torch.compile 对模型结构进行编译优化 (PyTorch >= 2.0)
    # ----------------------------------------------------
    # 它是 PyTorch 2.x 的核心加速技术：能够融合 CUDA 算子并移除 Python 解释器的循环开销。
    # 由于目前 Windows 系统官方对编译所需的 Triton 后端支持不佳，为避免 TritonMissing 报错，
    # 仅在非 Windows 环境且 CUDA 显卡可用时才自动开启模型编译。
    if hasattr(torch, 'compile') and torch.cuda.is_available() and not sys.platform.startswith('win'):
        try:
            print('正在使用 torch.compile 编译优化网络结构...')
            encoder = torch.compile(encoder)
            decoder = torch.compile(decoder)
        except Exception as e:
            print('编译优化失败，回退到原始模型模式：{}'.format(e))

    # ----------------------------------------------------
    # 7. 初始化自动混合精度 (AMP) 梯度缩放器 (GradScaler)
    # ----------------------------------------------------
    # 只有当 GPU 可用时，才实例化 Scaler。若处于 CPU 模式下，设为 None 自动回退传统 FP32 运算。
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    # 初始化计时和计量指标记录工具
    print('数据初始化就绪...')
    batch_time = AverageMeter()  
    losses = AverageMeter()  

    # ----------------------------------------------------
    # 8. 开启 Epoch 大循环
    # ----------------------------------------------------
    epoch = epoch_start
    try:
        for epoch in range(epoch_start, epochs):
            # 计划采样 (Scheduled Sampling)：
            # 随着训练进行，模型逐渐变聪明，应当减少导师指导的比例。
            # 前 30 个 Epoch，导师迫导比例从 0.8 线性衰减至 0.1；30 个 Epoch 之后固定为 0.1，促使模型完全独立学习并消除暴露偏差。
            global teacher_forcing_ratio
            teacher_forcing_ratio = max(0.1, 0.8 - epoch * (0.7 / 30.0))
            print('\n--- Epoch {} 开始，当前教师迫导比例 (Teacher Forcing Ratio): {:.3f} ---'.format(epoch, teacher_forcing_ratio))

            # 将模型调整为训练状态（激活 Dropout 等随机层）
            encoder.train()
            decoder.train()

            # 预计算批次总数的字符宽度，用于日志对齐（例如 554 为 3 位，确保 0/554 与 100/554 列对齐）
            batch_width = len(str(len(train_loader)))

            # 遍历训练集 Batch 列表 (使用高效迭代的 DataLoader)
            for i, batch in enumerate(train_loader):
                start = time.time()  # 获取数据后才开始计时，以剥离 DataLoader 后台读取/重置等额外开销
                input_variable, lengths, target_variable, mask, max_target_len = batch
                
                # 执行单步批训练，获得当前损失值
                loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder,
                             encoder_optimizer, decoder_optimizer, scaler=scaler)

                # 记录指标
                losses.update(loss, max_target_len)
                batch_time.update(time.time() - start)

                # 按配置规律打印训练日志（使用右对齐填充确保批次索引列宽一致）
                if i % print_every == 0:
                    print('[{0}] Epoch: [{1}][{2:>{width}}/{3}]\t'
                          '批处理耗时 {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          '交叉熵 Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                              timestamp(), epoch, i, len(train_loader),
                              width=batch_width,
                              batch_time=batch_time,
                              loss=losses))
            
            # 每个 Epoch 结束后，在验证集上计算一次真实评估损失
            val_loss = validate(val_loader, encoder, decoder)
            print('\n * 验证集损失 - {loss:.3f}\n'.format(loss=val_loss))

            # 自动展示：随机抽取 10 句验证集句子让模型作答，直观评估当前模型翻译/对话水平
            # 如果模型被 compile 编译过，我们需要解包底层原始模型 _orig_mod 来进行搜索预测，以防编译带来的推理延迟
            searcher = GreedySearchDecoder(getattr(encoder, '_orig_mod', encoder), getattr(decoder, '_orig_mod', decoder))
            for sentence in pick_n_valid_sentences(10):
                decoded_words = evaluate(searcher, sentence)
                print('人类发问: {}'.format(sentence))
                print('机器人作答: {}'.format(''.join(decoded_words)))

            # 定期自动保存 Checkpoint 文件
            if epoch % save_every == 0:
                directory = save_dir
                if not os.path.exists(directory):
                    os.makedirs(directory)
                torch.save({
                    'epoch': epoch,
                    'en': getattr(encoder, '_orig_mod', encoder).state_dict(),
                    'de': getattr(decoder, '_orig_mod', decoder).state_dict(),
                    'en_opt': encoder_optimizer.state_dict(),
                    'de_opt': decoder_optimizer.state_dict(),
                    'loss': loss,
                    'voc': voc.__dict__
                }, os.path.join(directory, '{}_{}_{}.tar'.format('checkpoint', epoch, val_loss)))
    except KeyboardInterrupt:
        # 优雅退出与现场保护：如果用户手动按 Ctrl+C 中断，自动将当前训练现场转储，防止进度丢失
        print('\n[检测到键盘中断 (Ctrl+C)] 正在优雅退出...')
        directory = save_dir
        if not os.path.exists(directory):
            os.makedirs(directory)
        checkpoint_path = os.path.join(directory, 'checkpoint_interrupted.tar')
        torch.save({
            'epoch': epoch,
            'en': getattr(encoder, '_orig_mod', encoder).state_dict(),
            'de': getattr(decoder, '_orig_mod', decoder).state_dict(),
            'en_opt': encoder_optimizer.state_dict(),
            'de_opt': decoder_optimizer.state_dict(),
            'loss': losses.avg,
            'voc': voc.__dict__
        }, checkpoint_path)
        print('当前训练进度已优雅保存至: {}'.format(checkpoint_path))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', '-c', type=str, default='xhj', choices=['xhj', 'ptt'], help='选择训练的语料: xhj (小黄鸡) 或 ptt (PTT语料)')
    args = parser.parse_args()
    main()
