#!/usr/bin/env bash

# Copyright 2024 Michael Ong

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


if [ $# -ne 1 ]; then
    echo "Usage: $0 bench_corpus_dir"
    exit 1
fi

set -eo pipefail

# splits text into individual phones + turns spaces into _
function split_text () {
  text="$1"

  awk \
  'BEGIN {
    FS = " ";
    OFS = " ";
  }

  {
    gsub(/ /, "_");
    gsub(/\[fp\]|d[zʒ]ː|tʃː|d[zʒ]|tʃ|\Sː|\S/, "& ");
    gsub(/ +/, " ");
    print $0;
  }' "$text"
}

bench_dir="$1"
out_dir="data/lstm"
partitions=(train test dev)

for partition in "${partitions[@]}"; do
    mkdir -p "$out_dir"/"$partition"/
    :> "$out_dir"/"$partition"/wav.scp
    :> "$out_dir"/"$partition"/trn
    :> "$out_dir"/"$partition"/trn_lstm
    for file in "$bench_dir"/"$partition"/*.wav; do
        filename="$(basename "$file" .wav)"
        printf "%s %s\n" "$filename" "$file" >> "$out_dir"/"$partition"/wav.scp
        printf "%s (%s)\n" "$(< "${file%%.wav}.txt")" "$filename" >> "$out_dir"/"$partition"/trn
        printf "%s(%s)\n" "$(split_text "${file%%.wav}.txt")" "$filename" >> "$out_dir"/"$partition"/trn_lstm
    done

    if [[ $partition == "train" ]]; then
        awk '{NF --; print}' "$out_dir"/train/trn_lstm |
        tr ' ' $'\n' |
        sort -u |
        awk 'NF != 0' |
        awk '{print $0,NR-1}' > "$out_dir"/token2id
    fi

    signals-to-torch-feat-dir \
    "$out_dir"/"$partition"/wav.scp prep/conf/feats/fbank_41.json "$out_dir"/"$partition"/feat/
    trn-to-torch-token-data-dir \
    "$out_dir"/"$partition"/trn_lstm "$out_dir"/token2id "$out_dir"/"$partition"/ref/
    num_filts="$(awk 'NR == 3 {print $2}' <<< "$(get-torch-spect-data-dir-info --strict "$out_dir"/"$partition")")"
    if [[ num_filts -ne 41 ]]; then
        echo "$partition has incorrect feature dims"
        exit 1
    fi
done


