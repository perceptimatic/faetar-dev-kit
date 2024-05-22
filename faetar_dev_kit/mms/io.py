# Copyright 2024 Sean Robertson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import json

from re import compile

from tqdm import tqdm

from csv import DictReader
from collections import Counter

import jiwer

from .args import Options


def compile_metadata(options: Options):

    fp = (options.data / "metadata.csv").open("w")
    fp.write("file_name,sentence\n")

    for wav in tqdm(sorted(options.data.glob("*.wav")),
                    f"Processing directory {options.data}"):
        entries = [wav.name]
        if not options.no_sentence:
            txt = options.bench / (wav.stem + ".txt")
            if not txt.is_file():
                print(
                    f"'{wav}' exists, but '{txt}' does not! If you don't want "
                    "transcripts, add the --no-sentence flag",
                    file=sys.stderr,
                )
                return 1
            entries.append(txt.read_text().strip())
        fp.write(",".join(entries))
        fp.write("\n")

    return 0


def write_vocab(options: Options):

    if options.append:
        with options.vocab_json.open() as fp:
            vocab_json = json.load(fp)
    else:
        vocab_json = dict()

    csv = DictReader(options.metadata_csv.open(newline=""), delimiter=",")

    vocab2count = Counter()
    for no, row in enumerate(csv):
        for word in row["sentence"].strip().split():
            if word.startswith("["):
                if not word.endswith("]"):
                    print(
                        f"found invalid token '{word}' in line {no + 2} of "
                        f"'{options.metadata_csv}'!",
                        file=sys.stderr,
                    )
                    return 1
                word = (word,)
            vocab2count.update(word)

    if options.pad in vocab2count:
        print(
            f"--pad token '{options.pad}' found in '{options.metadata_csv}'!",
            file=sys.stderr,
        )
        return 1

    if options.word_delimiter in vocab2count:
        print(
            f"--word-delimiter token '{options.word_delimiter}' found in "
            f"'{options.metadata_csv}'!",
            file=sys.stderr,
        )
        return 1

    if options.unk in vocab2count:
        print(
            f"--unk token '{options.unk}' found in '{options.metadata_csv}'. "
            "This could be intentional",
            file=sys.stderr,
        )
        del vocab2count[options.unk]

    vocab = sorted(
        vocab for (vocab, count) in vocab2count.items() if count > options.prune_count
    )
    del vocab2count

    # always store last in fixed order
    vocab.append(options.word_delimiter)
    vocab.append(options.unk)
    vocab.append(options.pad)

    vocab_json[options.lang] = dict((k, v) for (v, k) in enumerate(vocab))
    del vocab

    json.dump(vocab_json, options.vocab_json.open("w"))

    return 0

def metadata_to_trn(options: Options):

    trn = [
        (os.path.splitext(os.path.basename(row["file_name"]))[0], row["sentence"])
        for row in DictReader(options.metadata_csv.open(newline=""), delimiter=",")
    ]
    trn.sort()

    fp = options.trn.open("w")
    for utt, transcript in trn:
        fp.write(f"{transcript} ({utt})\n")

    return 0


def vocab_to_token2id(options: Options):

    token2id = json.load(options.vocab_json.open())[options.lang]

    fp = options.token2id.open("w")
    for token, id_ in sorted(token2id.items()):
        fp.write(f"{token} {id_}\n")

    return 0
