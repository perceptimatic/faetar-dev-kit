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
    cp "$conf_dir"/* "$exp_dir"/"$out_dir"/conf/
fi

# fixes vocab size
awk -v number="$(wc -l < "$data_dir"/prep/token2id)" \
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
    if [ ! -f ""$exp_dir"/"$out_dir"/trn_dec_"$x"" ]; then
        # no lm
        python3 prep/asr-baseline.py \
        --read-model-yaml "$exp_dir"/"$out_dir"/conf/model.yaml \
        decode \
        --read-data-yaml "$exp_dir"/"$out_dir"/conf/data.yaml \
        "$exp_dir"/"$out_dir"/best.ckpt "$data_dir"/"$x" "$exp_dir"/"$out_dir"/hyp_"$x"

        torch-token-data-dir-to-trn \
        "$exp_dir"/"$out_dir"/hyp_"$x" --swap "$data_dir"/prep/token2id "$exp_dir"/"$out_dir"/trn_dec_"$x"
    fi

    for y in per cer wer; do
      if [ ! -f ""$exp_dir"/"$out_dir"/error_report_eval_"$x"_"$y"" ]; then
        tempfile1="$(mktemp)"
        tempfile2="$(mktemp)"

        awk 'BEGIN {print "file_name,sentence"} {gsub(/[\(\)]/, "", $NF); filename=$NF; NF--; print filename","$0}' "$data_dir"/prep/trn_"$x" > "$tempfile1"
        awk 'BEGIN {print "file_name,sentence"} {gsub(/[\(\)]/, "", $NF); filename=$NF; NF--; print filename","$0}' "$exp_dir"/"$out_dir"/trn_dec_"$x" > "$tempfile2"

        python3 ./mms.py evaluate --error-type "$y" \
        "$tempfile1" "$tempfile2" > "$exp_dir"/"$out_dir"/error_report_eval_"$x"_"$y"
      fi
    done
    
    # if [ ! -f ""$exp_dir"/"$out_dir"/error_report_"$x"" ]; then
    #     python3 prep/error-rates-from-trn.py \
    #     "$data_dir"/prep/trn_"$x" "$exp_dir"/"$out_dir"/trn_dec_"$x" > "$exp_dir"/"$out_dir"/error_report_"$x"
    # fi
done