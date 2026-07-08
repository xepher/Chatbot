import os
import sys
import torch
import torch.nn as nn
import jieba
import argparse

from config import *
from models import EncoderRNN, LuongAttnDecoderRNN
from utils import GreedySearchDecoder, indexesFromSentence, find_checkpoint

def main():
    # 使用 argparse 来解析命令行参数，支持自定义选择 Checkpoint
    parser = argparse.ArgumentParser(description='交互式对话客户端')
    parser.add_argument('--checkpoint', '-cp', type=str, default=None,
                        help='指定加载的 Checkpoint 权重文件名或路径。若不指定，则按默认规则自动选择。')
    args = parser.parse_args()

    # 1. 定位 Checkpoint 权重文件
    resolved_path = None
    
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            resolved_path = args.checkpoint
            print('已手动指定并加载 Checkpoint: {}'.format(resolved_path))
        else:
            # 尝试在 save_dir 目录下寻找指定文件
            possible_path = os.path.join(save_dir, args.checkpoint)
            if os.path.exists(possible_path):
                resolved_path = possible_path
                print('在保存目录下找到并加载 Checkpoint: {}'.format(resolved_path))
            else:
                print('错误：指定的 Checkpoint 权重文件不存在：{}'.format(args.checkpoint))
                sys.exit(1)
    else:
        # 默认自动选择逻辑：优先中断备份，其次最新保存模型，最后最优模型
        interrupted_path = os.path.join(save_dir, 'checkpoint_interrupted.tar')
        if os.path.exists(interrupted_path):
            resolved_path = interrupted_path
            print('正在自动加载上次中断的备份 Checkpoint: {}'.format(resolved_path))
        else:
            resolved_path = find_checkpoint(save_dir, mode='latest')
            if resolved_path:
                print('已自动选择最新保存的 Checkpoint: {}'.format(resolved_path))
            else:
                resolved_path = find_checkpoint(save_dir, mode='best')
                if resolved_path:
                    print('已自动选择当前最优的 Checkpoint: {}'.format(resolved_path))

    if not resolved_path or not os.path.exists(resolved_path):
        print('未能在 {} 目录下找到任何有效的 Checkpoint 权重文件，请先运行 train.py 训练模型。'.format(save_dir))
        sys.exit(1)

    print('正在从本地加载权重: {} ...'.format(resolved_path))
    checkpoint = torch.load(resolved_path, map_location=device)
    
    # 自动恢复词表映射
    if 'voc' in checkpoint:
        voc.__dict__.update(checkpoint['voc'])
    
    # 2. 实例化模型与层
    embedding = nn.Embedding(voc.num_words, hidden_size)
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

    # 3. 加载状态字典 (直接提取底层底层权重，无惧 compile 包装)
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])

    # 送入指定设备并设为 eval 推理模式
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.eval()
    decoder.eval()

    # 4. 实例化贪心解码推理器
    searcher = GreedySearchDecoder(encoder, decoder)

    print('\n======================================================')
    print('🤖 智能对话机器人加载就绪！输入 "exit" 或 "quit" 可退出对话。')
    print('======================================================\n')

    # 5. 交互式对话循环
    while True:
        try:
            # 获取人类输入
            sentence = input('人类发问 (You) > ')
            sentence = sentence.strip()
            if not sentence:
                continue
            if sentence.lower() in ('exit', 'quit'):
                print('对话结束，下回见！')
                break

            # 转换为词 ID 序列 (若遇到词表外未知词会抛出 KeyError)
            try:
                indexes_batch = [indexesFromSentence(voc, sentence)]
            except KeyError:
                print('机器人作答 (Bot) > 听到了一些我没学过的生疏词汇，换个更简单的说法试试吧 ╮(﹀_﹀)╭')
                print()
                continue

            lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
            input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)

            # 送往推理设备
            input_batch = input_batch.to(device)
            lengths = lengths.to(device)

            # 自回归推理计算，得到预测的单词 Token 序列
            tokens, scores = searcher(input_batch, lengths, max_len)

            # 解码回字符，忽略填充符与结束符
            decoded_words = [voc.index2word[token.item()] for token in tokens
                             if token != EOS_token and token != PAD_token]
            
            response = "".join(decoded_words)
            print('机器人作答 (Bot) > {}'.format(response))
            print()

        except KeyboardInterrupt:
            print('\n对话强行结束。再见！')
            break
        except Exception as e:
            print('发生未知错误: {}'.format(e))
            break

if __name__ == '__main__':
    main()
