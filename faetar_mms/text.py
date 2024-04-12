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

import sys

from csv import DictReader
from collections import Counter
from json import dump

from .args import Options


def write_vocab(options: Options):

    csv = DictReader(options.metadata_csv.open())

    vocab2count = Counter()
    for row in csv:
        vocab2count.update(row["sentence"].strip().split())

    if options.pad in vocab2count:
        print(
            f"--pad token '{options.pad}' found in {options.metadata_csv}!",
            file=sys.stderr,
        )
        return

    if options.unk in vocab2count:
        print(
            f"--unk token '{options.unk}' found in {options.metadata_csv}. "
            "This could be intentional",
            file=sys.stderr,
        )
        del vocab2count[options.unk]

    vocab = sorted(
        vocab for (vocab, count) in vocab2count.items() if count > options.prune_count
    )
    del vocab2count
    vocab.append(options.unk)
    vocab.append(options.pad)

    vocab_json = {options.iso: dict((k, v) for (v, k) in enumerate(vocab))}
    del vocab

    dump(vocab_json, options.vocab_json.open("w"))
