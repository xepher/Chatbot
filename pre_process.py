# encoding=utf-8
from collections import Counter
import jieba
from tqdm import tqdm
from config import *


def encode_text(word_map, c):
    return [word_map.get(word, word_map['<unk>']) for word in c] + [word_map['<end>']]


def build_wordmap(sentences):
    word_freq = Counter()

    for sentence in tqdm(sentences):
        seg_list = jieba.cut(sentence)
        # Update word frequency
        word_freq.update(list(seg_list))

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 4 for v, k in enumerate(words)}
    word_map['<pad>'] = 0
    word_map['<start>'] = 1
    word_map['<end>'] = 2
    word_map['<unk>'] = 3
    print(len(word_map))
    print(words[:10])

    return word_map


def build_samples(sentences, word_map):
    print('building samples')
    samples = []
    for i in tqdm(range(0, len(sentences) - 1, 2)):
        sentence_in = sentences[i]
        seg_list = jieba.cut(sentence_in)
        tokens_in = encode_text(word_map, list(seg_list))
        sentence_out = sentences[i + 1]
        seg_list = jieba.cut(sentence_out)
        tokens_out = encode_text(word_map, list(seg_list))
        if len(tokens_in) <= max_len and len(tokens_out) <= max_len and UNK_token not in (tokens_in + tokens_out):
            samples.append({'input': list(tokens_in), 'output': list(tokens_out)})

    return samples


def build_sentences_xhj():
    with open(xhj_corpus_loc, 'r', encoding="utf8") as f:
        sentences = f.readlines()
    print('total lines: ' + str(len(sentences)))

    sentences = [s[2:].strip() for s in sentences if len(s[1:].strip()) > 0]
    print(len(sentences))
    print('removed empty lines: ' + str(len(sentences)))

    return sentences


def parse_ptt_lines(lines):
    result = []
    for s in lines:
        s = s.strip()
        if not s:
            continue
        parts = s.split('\t')
        if len(parts) == 2:
            q = parts[0].strip()
            a = parts[1].strip()
            if q and a:
                result.append(q)
                result.append(a)
    return result


def build_sentences_ptt():
    with open(ptt_corpus_loc, 'r', encoding="utf8") as f:
        lines = f.readlines()
    print('total lines: ' + str(len(lines)))

    result = parse_ptt_lines(lines)
    print(len(result))
    print('removed empty lines and invalid QA pairs: ' + str(len(result)))

    return result


def save_data(data, filename):
    with open(filename, 'w', encoding="utf8") as f:
        json.dump(data, f, indent=4)
    print('{} data created at: {}.'.format(len(data), filename))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', '-c', type=str, default='xhj', choices=['xhj', 'ptt'], help='Choose corpus: xhj or ptt')
    args = parser.parse_args()

    if args.corpus == 'ptt':
        sentences = build_sentences_ptt()
    else:
        sentences = build_sentences_xhj()

    word_map = build_wordmap(sentences)
    save_data(word_map, wordmap_loc)
    samples = build_samples(sentences, word_map)
    save_data(samples, samples_loc)

