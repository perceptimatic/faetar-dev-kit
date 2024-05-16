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

usage="Usage: $0 [-h] [-w NAT] [-a NAT] [-B NAT] [-l NNINT] data-dir out-dir"
data_dir="$1"
out_dir="$2"
conf_dir="conf/lstm"
exp_dir="exp/lstm"
width=100
alpha_inv=1
beta=1
lm_ord=0
name=
help="Train and decode with the lstm baseline

Options
    -h          Display this help message and exit
    -w NAT      pyctcdecode's beam width (default: $width)
    -a NAT      pyctcdecode's alpha, inverted (default: $alpha_inv)
    -B NAT      pyctcdecode's beta, inverted (default: $beta)
    -l NAT      n-gram LM order. 0 is greedy; 1 is prefix with no LM (default: $lm_ord)"

while getopts "hw:a:B:l:" name; do
    case $name in
        h)
            echo "$usage"
            echo ""
            echo "$help"
            exit 0;;
        w)
            width="$OPTARG";;
        a)
            alpha_inv="$OPTARG";;
        B)
            beta="$OPTARG";;
        l)
            lm_ord="$OPTARG";;
        *)
            echo -e "$usage"
            exit 1;;
    esac
done
shift $(($OPTIND - 1))
if ! [ "$width" -gt 0 ] 2> /dev/null; then
    echo -e "$width is not a natural number! set -w appropriately!"
    exit 1
fi
if ! [ "$alpha_inv" -gt 0 ] 2> /dev/null; then
    echo -e "$alpha_inv is not a natural number! set -a appropriately!"
    exit 1
fi
if ! [ "$beta" -gt 0 ] 2> /dev/null; then
    echo -e "$beta is not a natural number! set -B appropriately!"
    exit 1
fi
if ! [ "$lm_ord" -ge 0 ] 2> /dev/null; then
    echo -e "$lm_ord is not a non-negative int! set -l appropriately!"
    exit 1
fi

if [ $# -ne 2 ]; then
  echo "$usage"
  exit 1
fi

if [ ! -d "$1" ]; then
  echo "$0: '$1' is not a directory"
  exit 1
fi

set -eo pipefail

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

mkdir -p "$exp_dir"/"$out_dir"/decode/

for x in test dev train; do
  if [ ! -f ""$exp_dir"/"$out_dir"/decode/"$x"_"$name".trn" ]; then

    tempfile=$(mktemp)

    if [ "$lm_ord" -eq 0 ]; then
      name="greedy"
      
      if [ ! -f "$exp_dir"/"$out_dir"/decode/hyp_"$x"_"$name".done ]; then
        python3 prep/asr-baseline.py \
          --read-model-yaml "$exp_dir"/"$out_dir"/conf/model.yaml \
          decode \
          --read-data-yaml "$exp_dir"/"$out_dir"/conf/data.yaml \
          "$exp_dir"/"$out_dir"/best.ckpt "$data_dir"/"$x" "$exp_dir"/"$out_dir"/decode/hyp_"$x"_"$name"
        
        touch "$exp_dir"/"$out_dir"/decode/hyp_"$x"_"$name".done
      fi

      torch-token-data-dir-to-trn \
        "$exp_dir"/"$out_dir"/decode/hyp_"$x"_"$name" --swap "$data_dir"/token2id "$tempfile"

    else
      if [ "$lm_ord" -eq 1 ]; then
        name="w${width}_nolm"
        alpha_inv=1
        beta=1
        lm_args=( )
      else
        name="w${width}_lm${lm_ord}_ainv${alpha_inv}_b${beta}"
        lm="$exp_dir/$out_dir/lm/${lm_ord}gram.arpa"
        lm_args=( --lm "$lm" )
        if ! [ -f "$lm" ]; then
          echo "Constructing '$lm'"
          mkdir -p "$exp_dir/$out_dir/lm"
          python3 prep/ngram_lm.py -o $lm_ord -t 0 1 < "etc/lm_text.txt" > "${lm}_"
          mv "$lm"{_,}
        fi
      fi

      if [ ! -f "$exp_dir"/"$out_dir"/decode/hyp_logits_"$x"_"$name".done ]; then
        python3 prep/asr-baseline.py \
          --read-model-yaml "$exp_dir"/"$out_dir"/conf/model.yaml \
          decode \
          --read-data-yaml "$exp_dir"/"$out_dir"/conf/data.yaml \
          --write-logits \
          "$exp_dir"/"$out_dir"/best.ckpt "$data_dir"/"$x" "$exp_dir"/"$out_dir"/decode/hyp_logits_"$x"_"$name"
        
        touch "$exp_dir"/"$out_dir"/decode/hyp_logits_"$x"_"$name".done
      fi

      python3 logits-to-trn-via-pyctcdecode.py \
        --char "${lm_args[@]}" \
        --words "etc/lm_words.txt" \
        --width $width \
        --beta $beta \
        --alpha-inv $alpha_inv \
        --token2id "$data_dir"/token2id \
        "$exp_dir"/"$out_dir"/decode/hyp_logits_"$x"_"$name" "$tempfile"
    fi

    # merges phones back into words
    awk \
    '{
      file = $NF;
      NF --;
      gsub(/ /, "");
      gsub(/_/, " ");
      gsub(/ +/, " ");
      print $0 " " file;
    }' "$tempfile" > "$exp_dir"/"$out_dir"/decode/"$x"_"$name".trn
  fi

  for y in per cer wer; do
    if [ ! -f ""$exp_dir"/"$out_dir"/decode/error_report_eval_"$x"_"$y"" ]; then
       ./evaluate_asr.sh -d "$data_dir" -e "$exp_dir" -p "$x" -r "$y" > "$exp_dir"/"$out_dir"/decode/error_report_eval_"$x"_"$y"
    fi
  done
done