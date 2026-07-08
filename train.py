# encoding=utf-8
import os
import random
import time
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from data_gen import ChatbotDataset, batch2TrainData
from models import EncoderRNN, LuongAttnDecoderRNN
from utils import *

def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, encoder_optimizer,
          decoder_optimizer, scaler=None):
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
    - scaler: AMP 梯度缩放器
    
    返回:
    - 本批次的平均每词交叉熵损失值 (Average loss per token)
    """
    # ----------------------------------------------------
    # 步骤 1. 梯度归零
    # ----------------------------------------------------
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
    # 步骤 3-6. 使用 autocast 开启自动混合精度前向计算
    # ----------------------------------------------------
    with autocast(enabled=torch.cuda.is_available()):
        # 步骤 3. 前向传播：编码问句 (Encoder)
        encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

        # 步骤 4. 初始化解码器输入与初隐状态 (Decoder Init)
        decoder_input = torch.LongTensor([[SOS_token for _ in range(input_variable.size(1))]])
        decoder_input = decoder_input.to(device)
        decoder_hidden = encoder_hidden[:decoder.n_layers]

        # 步骤 5. 教师迫导 (Teacher Forcing) 策略决策
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        # 步骤 6. 循环解码计算 Loss
        if use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                decoder_input = target_variable[t].view(1, -1)
                
                mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal
        else:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(input_variable.size(1))]])
                decoder_input = decoder_input.to(device)
                
                mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal

    # ----------------------------------------------------
    # 步骤 7. 反向传播与梯度更新 (支持 AMP)
    # ----------------------------------------------------
    if scaler is not None:
        scaler.scale(loss).backward()
        # 梯度剪裁前需要先 unscale
        scaler.unscale_(encoder_optimizer)
        scaler.unscale_(decoder_optimizer)
        _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
        scaler.step(encoder_optimizer)
        scaler.step(decoder_optimizer)
        scaler.update()
    else:
        loss.backward()
        _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
        encoder_optimizer.step()
        decoder_optimizer.step()

    # 返回该批次中每个非填充字符的平均 Loss 损失值
    return sum(print_losses) / n_totals


def validate(val_loader, encoder, decoder):
    """
    在验证集 (Validation Set) 上计算损失。
    验证过程中，只执行前向传播，不计算梯度，也不更新参数。
    """
    with torch.no_grad():
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

            # 前向计算使用 autocast
            with autocast(enabled=torch.cuda.is_available()):
                encoder_outputs, encoder_hidden = encoder(input_variable, lengths)
                
                decoder_input = torch.LongTensor([[SOS_token for _ in range(input_variable.size(1))]])
                decoder_input = decoder_input.to(device)
                decoder_hidden = encoder_hidden[:decoder.n_layers]

                for t in range(max_target_len):
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden, encoder_outputs
                    )
                    _, topi = decoder_output.topk(1)
                    decoder_input = torch.LongTensor([[topi[i][0] for i in range(input_variable.size(1))]])
                    decoder_input = decoder_input.to(device)
                    
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
    单步预测推理函数。用于测试或者在交互界面中回答用户的提问。
    """
    indexes_batch = [indexesFromSentence(voc, sentence)]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    
    tokens, scores = searcher(input_batch, lengths, max_length)
    
    decoded_words = [voc.index2word[token.item()] for token in tokens
                     if token != EOS_token and token != PAD_token]
    return decoded_words


def main():
    # 1. 实例化 PyTorch 的 Dataset 和加载器
    # 采用异步多进程并行 DataLoader 提速
    train_loader = DataLoader(
        ChatbotDataset('train'), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=workers, 
        collate_fn=batch2TrainData, 
        pin_memory=True
    )
    val_loader = DataLoader(
        ChatbotDataset('valid'), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=workers, 
        collate_fn=batch2TrainData, 
        pin_memory=True
    )

    # 2. 初始化嵌入矩阵权重（Embedding）。
    embedding = nn.Embedding(voc.num_words, hidden_size)

    # 3. 初始化 Encoder 和 Decoder 神经网络模型
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

    # 将网络参数复制到指定计算设备中
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # 4. 初始化 Adam 优化器 (Optimizers)。
    print('正在创建 Adam 优化器...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

    # 5. 加载 Checkpoint 以便断点续训
    epoch_start = start_epoch
    if loadFilename:
        resolved_path = loadFilename
        is_interrupted_cp = False

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
            epoch_start = checkpoint['epoch'] + 1
            encoder.load_state_dict(checkpoint['en'])
            decoder.load_state_dict(checkpoint['de'])
            encoder_optimizer.load_state_dict(checkpoint['en_opt'])
            decoder_optimizer.load_state_dict(checkpoint['de_opt'])
            if 'voc' in checkpoint:
                voc.__dict__.update(checkpoint['voc'])
            print('Checkpoint 加载成功，将从 Epoch {} 继续训练。'.format(epoch_start))

            if is_interrupted_cp:
                try:
                    os.remove(resolved_path)
                    print('已成功清理临时中断恢复包。')
                except OSError as e:
                    print('清理临时中断恢复包失败: {}'.format(e))
        elif resolved_path:
            print('警告: 指定的 Checkpoint 文件未找到，将从头开始训练: {}'.format(resolved_path))

    # 5.1 使用 torch.compile 对模型结构进行编译优化 (PyTorch >= 2.0 且在 GPU 下生效)
    # 由于 Windows 平台尚未原生支持 Triton，为了避免 TritonMissing 报错，在 Windows 系统下默认跳过编译
    if hasattr(torch, 'compile') and torch.cuda.is_available() and not sys.platform.startswith('win'):
        try:
            print('正在使用 torch.compile 编译优化网络结构...')
            encoder = torch.compile(encoder)
            decoder = torch.compile(decoder)
        except Exception as e:
            print('编译优化失败，回退到原始模型模式：{}'.format(e))

    # 5.2 初始化自动混合精度 (AMP) 梯度缩放器
    scaler = GradScaler() if torch.cuda.is_available() else None

    # 初始化计时和计量指标记录工具
    print('数据初始化就绪...')
    batch_time = AverageMeter()  
    losses = AverageMeter()  

    # 6. 开启 Epoch 大循环
    epoch = epoch_start
    try:
        for epoch in range(epoch_start, epochs):
            encoder.train()
            decoder.train()

            start = time.time()

            # 遍历训练集 Batch 列表 (使用 DataLoader)
            for i, batch in enumerate(train_loader):
                input_variable, lengths, target_variable, mask, max_target_len = batch
                
                loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder,
                             encoder_optimizer, decoder_optimizer, scaler=scaler)

                losses.update(loss, max_target_len)
                batch_time.update(time.time() - start)

                start = time.time()

                if i % print_every == 0:
                    print('[{0}] Epoch: [{1}][{2}/{3}]\t'
                          '批处理耗时 {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          '交叉熵 Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(timestamp(), epoch, i, len(train_loader),
                                                                           batch_time=batch_time,
                                                                           loss=losses))
            
            # 每个 Epoch 结束后计算一次验证集 Loss
            val_loss = validate(val_loader, encoder, decoder)
            print('\n * 验证集损失 - {loss:.3f}\n'.format(loss=val_loss))

            # 自动展示作答
            searcher = GreedySearchDecoder(getattr(encoder, '_orig_mod', encoder), getattr(decoder, '_orig_mod', decoder))
            for sentence in pick_n_valid_sentences(10):
                decoded_words = evaluate(searcher, sentence)
                print('人类发问: {}'.format(sentence))
                print('机器人作答: {}'.format(''.join(decoded_words)))

            # 保存模型权重 Checkpoint (确保排除 compile 包装)
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
