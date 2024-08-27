#! /usr/bin/env bash

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

export PYTHONUTF8=1
[ -f "path.sh" ] && . "path.sh"

usage="Usage: $0 [-h] [-o] [-e DIR] [-d DIR] [-m DIR] [-w NAT] [-a NAT] [-b NAT] [-l NNINT]"
only=false
exp=exp/mms_lsah
data=data/mms_lsah
model="facebook/mms-1b-all"
width=100
alpha_inv=1
beta=1
lm_ord=0
training_kwargs=conf/mms_lsah/training_kwargs.json
wav2vec2_kwargs=conf/mms_lsah/wav2vec2_kwargs.json
dec_partitions=(train dev test)
bootstrap_samples=0
help="Train and decode with the mms-lsah baseline

Options
    -h          Display this help message and exit
    -o          Run only the next step of the script
    -e DIR      The experiment directory (default: '$exp')
    -d DIR      The data directory (default: '$data')
    -m DIR      The path to the model, or Hugging Face id (default: '$model')
    -c FILE     Path to TrainingArguments JSON keyword args (default: '$training_kwargs')
    -C FILE     Path to Wav2Vec2Config JSON keyword args (default: '$wav2vec2_kwargs')
    -w NAT      pyctcdecode's beam width (default: $width)
    -a NAT      pyctcdecode's alpha, inverted (default: $alpha_inv)
    -B NAT      pyctcdecode's beta (default: $beta)
    -l NAT      n-gram LM order. 0 is greedy; 1 is prefix with no LM (default: $lm_ord)
    -s NAT      Bootstrap samples. 0 is no bootstrap (default: $bootstrap_samples)"

while getopts "hoe:d:m:c:C:w:a:B:l:s:" name; do
    case $name in
        h)
            echo "$usage"
            echo ""
            echo "$help"
            exit 0;;
        o)
            only=true;;
        e)
            exp="$OPTARG";;
        d)
            data="$OPTARG";;
	m)
	    model="$OPTARG";;
        c)
            training_kwargs="$OPTARG";;
        C)
            wav2vec2_kwargs="$OPTARG";;
        w)
            width="$OPTARG";;
        a)
            alpha_inv="$OPTARG";;
        B)
            beta="$OPTARG";;
        l)
            lm_ord="$OPTARG";;
        s)
            bootstrap_samples="$OPTARG";;
        *)
            echo -e "$usage"
            exit 1;;
    esac
done
shift $(($OPTIND - 1))
for part in train dev test; do
    if [ ! -d "$data/$part" ]; then
        echo -e "'$data/$part' is not a directory! set -d appropriately!"
        exit 1
    fi
done
if ! mkdir -p "$exp" 2> /dev/null; then
    echo -e "Could not create '$exp'! set -e appropriately!"
    exit 1
fi
if ! [ -f "$training_kwargs" ]; then
    echo -e "'$training_kwargs' is not a file! Set -c appropriately!"
    exit 1
fi
if ! [ -f "$wav2vec2_kwargs" ]; then
    echo -e "'$wav2vec2_kwargs' is not a file! Set -C appropriately!"
    exit 1
fi
if ! [ "$width" -gt 0 ] 2> /dev/null; then
    echo -e "$width is not a natural number! set -w appropriately!"
    exit 1
fi
if ! [ "$alpha_inv" -gt 0 ] 2> /dev/null; then
    echo -e "$alpha_inv is not a natural number! set -a appropriately!"
    exit 1
fi
if (( "$(bc -l <<< "$beta < 0")" )); then
    echo -e "$beta is not greater than 0! set -B appropriately!"
    exit 1
fi
if ! [ "$lm_ord" -ge 0 ] 2> /dev/null; then
    echo -e "$lm_ord is not a non-negative int! set -l appropriately!"
    exit 1
fi
if ! [ "$bootstrap_samples" -ge 0 ] 2> /dev/null; then
    echo -e "$bootstrap_samples is not a non-negative int! set -n appropriately!"
    exit 1
fi

set -eo pipefail

if [ ! -f "prep/ngram_lm.py" ]; then
    echo "Initializing Git submodule"
    git submodule update --init --remote prep
fi

for part in train dev test; do
    if ! [ -f "$data/$part/metadata.csv" ]; then
        echo "Creating metadata.csv in '$data/$part'"
        mkdir -p "$data/$part"
        ./mms.py compile-metadata "$data/$part"
        if $only; then exit 0; fi
    fi

    if ! [ -f "$data/$part/trn" ]; then
        echo "Creating reference trn file in '$data/$part'"
        :> "$data/$part/trn"
        for file in "$data"/"$part"/*.wav; do
            filename="$(basename "$file" .wav)"
            printf "%s (%s)\n" "$(< "${file%%.wav}.txt")" "$filename" >> "$data/$part/trn"
        done
        if $only; then exit 0; fi
    fi
done

if ! [ -f "$exp/vocab.json" ]; then
    echo "Creating $exp/vocab.json"
    ./mms.py write-vocab "$data/train/metadata.csv" "$exp/vocab.json"
    if $only; then exit 0; fi
fi

if ! [ -f "$exp/config.json" ]; then
    echo "Training model and writing to '$exp'"
    ./mms.py train "$exp/vocab.json" "$data/"{train,dev} "$exp" \
	    --pretrained-model-id="$model" \
            --training-kwargs-json="$training_kwargs" \
            --wav2vec2-kwargs-json="$wav2vec2_kwargs"

    if $only; then exit 0; fi
fi

if [ "$lm_ord" = 0 ]; then
    for part in "${dec_partitions[@]}"; do
        if  ! [ -f "$exp/decode/${part}_greedy.trn" ]; then
            echo "Greedily decoding '$data/$part'"
            mkdir -p "$exp/decode"
            ./mms.py decode \
                "$exp" "$data/$part" "$exp/decode/${part}_greedy.csv_"
            mv "$exp/decode/${part}_greedy.csv"{_,}
            ./mms.py metadata-to-trn \
                "$exp/decode/${part}_greedy."{csv,trn_}
            mv "$exp/decode/${part}_greedy.trn"{_,}
            if $only; then exit 0; fi
        fi
    done
else

    if [ ! -f "prep/ngram_lm.py" ]; then
        echo "Initializing Git submodule"
        git submodule update --init --remote prep
        if $only; then exit 0; fi
    fi

    for part in "${dec_partitions[@]}"; do
        if  ! [ -f "$exp/decode/logits/$part/.done" ]; then
            echo "Dumping logits of '$data/$part'"
            mkdir -p "$exp/decode/logits/$part"
            ./mms.py decode \
                --dump-logits "$exp/decode/logits/$part" \
                "$exp" "$data/$part" "/dev/null"
            touch "$exp/decode/logits/$part/.done"
            if $only; then exit 0; fi
        fi
    done

    if [ "$lm_ord" = 1 ]; then
        name="w${width}_nolm"
        alpha_inv=1
        beta=1
        lm_args=( )
    else
        name="w${width}_lm${lm_ord}_ainv${alpha_inv}_b${beta}"
        lm="$exp/lm/${lm_ord}gram.arpa"
        lm_args=( --lm "$lm" )
        if ! [ -f "$lm" ]; then
            echo "Constructing '$lm'"
            mkdir -p "$exp/lm"
            ./prep/ngram_lm.py -o $lm_ord -t 0 1 -f "etc/lm_text.txt" > "${lm}_"
            mv "$lm"{_,}
            if $only; then exit 0; fi
        fi
    fi

    if ! [ -f "$exp/token2id" ]; then
        echo "Constructing '$exp/token2id'"
        ./mms.py vocab-to-token2id "$exp/"{vocab.json,token2id}
        if $only; then exit 0; fi
    fi

    for part in "${dec_partitions[@]}"; do
        if ! [ -f "$exp/decode/${part}_${name}.trn" ]; then
            echo "Decoding $part"
            ./prep/logits-to-trn-via-pyctcdecode.py \
                --char "${lm_args[@]}" \
                --words "etc/lm_words.txt" \
                --width $width \
                --beta $beta \
                --alpha-inv $alpha_inv \
                --token2id "$exp/token2id" \
                "$exp/decode/"{logits/$part,${part}_$name.trn}
            if $only; then exit 0; fi
        fi
    done
fi

for er in wer cer per; do
    echo "===================================================================="
    echo "                       ERROR TYPE: $er                              "
    echo "===================================================================="
    echo ""
    for part in "${dec_partitions[@]}"; do
        ./evaluate_asr.sh -d "$data" -e "$exp" -p "$part" -r "$er" -n "$bootstrap_samples"
        echo ""
    done
done
