#! /usr/bin/env python

# Copyright 2024 Michael Ong
# Adapted from https://huggingface.co/learn/nlp-course/en/chapter6/5 and 
# https://medium.com/@varunsivamani/byte-pair-encoding-bpe-5fdced1b31cd

# Apache 2.0

import sys
import re

def get_word_counts(text):
    word_counts = dict()
    words = re.split(" ", re.sub(r'\n', " ", text))
    words = filter(None, words)
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    return (word_counts)

def phone_split(text):
    tokenized = re.sub(r'(\[fp\]|d[zʒ]ː|t[sʃ]ː|d[zʒ]|t[sʃ]|\Sː|\S)', r'\1 ', text)
    return (tokenized)

def get_pair_counts(word_counts, split_words):
    pair_counts = dict()
    for word in list(split_words.keys()):
        split_word = list(filter(None, re.split(" ", split_words[word])))
        word_count = word_counts[word]
        for i in range(len(split_word) - 1):
            pair = (split_word[i], split_word[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + word_count
    return pair_counts

def compress_splits(best_pair, split_words):
    compressed_splits = dict()
    for whole_word in split_words:
        compressed_word = re.sub(re.escape(best_pair[0] + " " + best_pair[1] + " "), re.escape(best_pair[0] + best_pair[1] + " "), split_words[whole_word])
        compressed_splits[whole_word] = compressed_word
    return compressed_splits


def main():
    text_file = sys.argv[1]
    out_path = sys.argv[2]

    with open(text_file, "r") as file:
        text = file.read()
        file.close()

    vocab = dict()
    word_counts = get_word_counts(text)
    split_words = {word: phone_split(word) for word in word_counts.keys()}
    pair_counts = get_pair_counts(word_counts, split_words)

    i = 0
    while len(vocab.keys()) < 200:
        best_pair = max(pair_counts, key=pair_counts.get)
        split_words = compress_splits(best_pair, split_words)
        pair_counts = get_pair_counts(word_counts, split_words)
        for word in split_words:
            tokens = re.split(" ", split_words[word])
            tokens = filter(None, tokens)
            for token in tokens:
                vocab[token] = vocab.get(token, 0) + word_counts[word]
        i += 1
        if i == 3:
            break

    # for word in split_words:
    #     tokens = re.split(" ", split_words[word])
    #     tokens = filter(None, tokens)
    #     for token in tokens:
    #         vocab[token] = vocab.get(token, 0) + word_counts[word]

    out = open(out_path, "w")

    # for word in list(word_counts):
    #     out.write(f"{word}\t{word_counts[word]}\n")
    # for word in list(split_words):
    #     out.write(f"{word}\t{split_words[word]}\n")
    # for pair in list(pair_counts):
    #     out.write(f"{pair}\t{pair_counts[pair]}\n")

    sorted_vocab = sorted(list((token, vocab[token]) for token in vocab), key=lambda x: x[1])
    for i in range(len(sorted_vocab) - 1, -1, -1):
        if sorted_vocab[i][0] == "[fp]":
            out.write(f"{sorted_vocab[i][0]}\t{sorted_vocab[i][0]}\n")
        else:
            out.write(f"{sorted_vocab[i][0]}\t{' '.join(sorted_vocab[i][0])}\n")
    # out.write("{")
    # for i in range(len(sorted_vocab) - 1, -1, -1):
    #     if i != 0:
    #         out.write(f"\"{sorted_vocab[i][0]}\": {sorted_vocab[i][1]}, ")
    #     else:
    #         out.write(f"\"{sorted_vocab[i][0]}\": {sorted_vocab[i][1]}")
    # out.write("}")
    out.close()

if __name__ == "__main__":
    sys.exit(main())
