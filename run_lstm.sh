#!/usr/bin/env bash

exp_dir="exp_lstm"

. utils/parse_options.sh

if [ $# -ne 2 ]; then
  echo "Usage: $0 data-dir out-dir"
  exit 1
fi

set -eo pipefail

if [ ! -d "$1" ]; then
  echo "$0: '$1' is not a directory"
  exit 1
fi

if [ ! -d "$2" ]; then
  echo "$0: '$2' is not a directory"
  exit 1
fi

data_dir="$1"
out_dir="$2"

mkdir -p "$exp_dir"/"$out_dir"

if [ ! -d ""$exp_dir"/"$out_dir"/conf" ]; then
    cp -r conf "$exp_dir"/"$out_dir"/
fi


awk -v number="$(awk '{number = 0; if ($2 > number) {number = $2}} END {print number + 1}' token2id)" \
'NR == 5 {print "vocab_size: " number} NR != 5 {print $0}' "$exp_dir"/"$out_dir"/conf/model.yaml > "$exp_dir"/"$out_dir"/conf/model.yaml_
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
for x in test train; do
    if [ ! -f ""$exp_dir"/"$out_dir"/error_report_"$x"" ]; then
        # no lm
        python3 prep/asr-baseline.py \
        --read-model-yaml "$exp_dir"/"$out_dir"/conf/model.yaml \
        decode \
        --read-data-yaml "$exp_dir"/"$out_dir"/conf/data.yaml \
        "$exp_dir"/"$out_dir"/best.ckpt "$data_dir"/"$x" "$exp_dir"/"$out_dir"/hyp_"$x"

        torch-token-data-dir-to-trn \
        "$exp_dir"/"$out_dir"/hyp_"$x" --swap token2id "$exp_dir"/"$out_dir"/trn_dec_"$x"

        python3 prep/error-rates-from-trn.py \
        trn_"$x" "$exp_dir"/"$out_dir"/trn_dec_"$x" > "$exp_dir"/"$out_dir"/error_report_"$x"
    fi
done
