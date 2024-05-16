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

exp_dir="exp/lstm"

if [ $# -ne 2 ]; then
  echo "Usage: $0 data-dir out-dir"
  exit 1
fi

set -eo pipefail

if [ ! -d "$1" ]; then
  echo "$0: '$1' is not a directory"
  exit 1
fi

data_dir="$1"
out_dir="$2"
conf_dir="conf/lstm"

mkdir -p "$exp_dir"/"$out_dir"

if [ ! -d ""$exp_dir"/"$out_dir"/conf" ]; then
    mkdir -p "$exp_dir"/"$out_dir"/conf
    cp "$conf_dir"/* "$exp_dir"/"$out_dir"/conf/
fi

# fixes vocab size
awk -v number="$(wc -l < "$data_dir"/token2id)" \
'NR == 5 {print "vocab_size: " number} NR != 5 {print $0}' "$exp_dir"/"$out_dir"/conf/model.yaml \
> "$exp_dir"/"$out_dir"/conf/model.yaml_
mv "$exp_dir"/"$out_dir"/conf/model.yaml{_,}

if [ ! -f ""$exp_dir"/"$out_dir"/best.ckpt" ]; then
    python3 prep/asr-baseline.py \
    --read-model-yaml "$exp_dir"/"$out_dir"/conf/model.yaml \
    train \
    --state-dir "$exp_dir"/"$out_dir"/ --state-csv "$exp_dir"/"$out_dir"/history.csv \
    --read-data-yaml "$exp_dir"/"$out_dir"/conf/data.yaml \
    --read-training-yaml "$exp_dir"/"$out_dir"/conf/training.yaml \
    "$data_dir"/train "$data_dir"/dev "$exp_dir"/"$out_dir"/best.ckpt
fi

for x in test dev train; do

    # no lm
    if [ ! -f ""$exp_dir"/"$out_dir"/decode/trn_dec_"$x"" ]; then
        python3 prep/asr-baseline.py \
        --read-model-yaml "$exp_dir"/"$out_dir"/conf/model.yaml \
        decode \
        --read-data-yaml "$exp_dir"/"$out_dir"/conf/data.yaml \
        "$exp_dir"/"$out_dir"/best.ckpt "$data_dir"/"$x" "$exp_dir"/"$out_dir"/hyp_"$x"

        mkdir -p "$exp_dir"/"$out_dir"/decode/

        torch-token-data-dir-to-trn \
        "$exp_dir"/"$out_dir"/hyp_"$x" --swap "$data_dir"/token2id "$exp_dir"/"$out_dir"/decode/trn_dec_"$x"
    fi

    if [ ! -f ""$exp_dir"/"$out_dir"/decode/"$x"_greedy.trn" ]; then
        # merges phones back into words
        awk \
        '{
          file = $NF;
          NF --;
          gsub(/ /, "");
          gsub(/_/, " ");
          gsub(/ +/, " ");
          print $0 " " file;
        }' "$exp_dir"/"$out_dir"/decode/trn_dec_"$x" > "$exp_dir"/"$out_dir"/decode/"$x"_greedy.trn
    fi

    for y in per cer wer; do
      if [ ! -f ""$exp_dir"/"$out_dir"/decode/error_report_eval_"$x"_"$y"" ]; then
         ./evaluate_asr.sh -d "$data_dir" -e "$exp_dir" -p "$x" -r "$y" > "$exp_dir"/"$out_dir"/decode/error_report_eval_"$x"_"$y"
      fi
    done
done