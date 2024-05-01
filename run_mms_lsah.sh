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

# XXX(sdrobert): keep up-to-date with faetar_mms/io.py:evaluate(...)
filter() {
    fn="$1"
    sed 's/\[[^]][^]]*\] //g; s/<[^>][^>]*> //g;' "$fn" > "${fn}_filt"
    ./prep/word2subword.py "$fn"{_filt,_char}
    sed 's/_ //g' "${fn}_char" > "${fn}_phone"
}


usage="Usage: $0 [-h] [-o] [-e DIR] [-d DIR] [-w NAT] [-a NAT] [-b NAT] [-l NNINT]"
only=false
exp=exp/mms_lsah
data=data/mms_lsah
width=100
alpha_inv=1
beta=1
lm_ord=1
training_kwargs=conf/mms_lsah/training_kwargs.json
wav2vec2_kwargs=conf/mms_lsah/wav2vec2_kwargs.json
bootstrap_samples=0
help="Train and evaluate the faetar-mms baseline

Options
    -h          Display this help message and exit
    -o          Run only the next step of the script
    -e DIR      The experiment directory (default: '$exp')
    -d DIR      The data directory (default: '$data')
    -c FILE     Path to TrainingArguments JSON keyword args (default: '$training_kwargs')
    -C FILE     Path to Wav2Vec2Config JSON keyword args (default: '$wav2vec2_kwargs')
    -w NAT      pyctcdecode's beam width (default: $width)
    -a NAT      pyctcdecode's alpha, inverted (default: $alpha_inv)
    -b NAT      pyctcdecode's beta, inverted (default: $beta)
    -l NAT      n-gram LM order. 1 is no LM (default: $lm_ord)
    -n NNINT    Bootstrap samples. 0 is no bootstrap (default: $bootstrap_samples)"

while getopts "hoe:d:c:C:a:b:l:n:" name; do
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
        c)
            training_kwargs="$OPTARG";;
        C)
            wav2vec2_kwargs="$OPTARG";;
        a)
            alpha_inv="$OPTARG";;
        b)
            beta="$OPTARG";;
        l)
            lm_ord="$OPTARG";;
        n)
            bootstrap_samples="$OPTARG";;
        *)
            echo -e "$usage"
            exit 1;;
    esac
done
shift $(($OPTIND - 1))
for d in "$data/"{train,dev,test}; do
    if [ ! -d "$d" ]; then
        echo -e "'$d' is not a directory! set -d appropriately!"
        exit 1
    fi
done
if ! [ -f "$training_kwargs" ]; then
    echo -e "'$training_kwargs' is not a file! Set -c appropriately!"
    exit 1
fi
if ! [ -f "$wav2vec2_kwargs" ]; then
    echo -e "'$wav2vec2_kwargs' is not a file! Set -C appropriately!"
    exit 1
fi
if ! mkdir -p "$exp" 2> /dev/null; then
    echo -e "Could not create '$exp'! set -e appropriately!"
    exit 1
fi
if ! [ "$alpha_inv" -gt 0 ] 2> /dev/null; then
    echo -e "$alpha_inv is not a natural number! set -a appropriately!"
    exit 1
fi
if ! [ "$beta" -gt 0 ] 2> /dev/null; then
    echo -e "$beta_inv is not a natural number! set -b appropriately!"
    exit 1
fi
if ! [ "$lm_ord" -gt 0 ] 2> /dev/null; then
    echo -e "$lm_ord is not a natural number! set -l appropriately!"
    exit 1
fi
if ! [ "$bootstrap_samples" -ge 0 ] 2> /dev/null; then
    echo -e "$bootstrap_samples is not a non-negative int! set -n appropriately!"
    exit 1
fi

set -e

if [ ! -f "prep/ngram_lm.py" ]; then
    echo "Initializing Git submodules"
    git submodule init
    git submodule update
fi

for d in "$data/"{train,dev,test}; do
    if ! [ -f "$d/metadata.csv" ]; then
        echo "Creating metadata.csv in '$d'"
        ./step.py compile-metadata "$d"
        if $only; then exit 0; fi
    fi
done

if ! [ -f "$exp/vocab.json" ]; then
    echo "Creating $exp/vocab.json"
    ./step.py write-vocab "$data/train/metadata.csv" "$exp/vocab.json"
    if $only; then exit 0; fi
fi

if ! [ -f "$exp/config.json" ]; then
    echo "Training model and writing to '$exp'"
    ./step.py train "$exp/vocab.json" "$data/"{train,dev} "$exp"
    if $only; then exit 0; fi
fi

for part in dev test; do
    if  ! [ -f "$exp/decode/${part}_greedy.trn_phone" ]; then
        echo "Greedy decoding and dumping logits of '$data/$part'"
        mkdir -p "$exp/decode/logits/$part"
        ./step.py decode \
                --logits-dir "$exp/decode/logits/$part" \
                "$exp" "$data/$part" "$exp/decode/${part}_greedy.csv"
        ./step.py metadata-to-trn "$exp/decode/${part}_greedy."{csv,trn}
        filter "$exp/decode/${part}_greedy.trn"
        if $only; then exit 0; fi
    fi
done

if ! [ -f "$exp/token2id" ]; then
    echo "Constructing '$exp/token2id'"
    ./step.py vocab-to-token2id "$exp/"{vocab.json,token2id}
    if $only; then exit 0; fi
fi


for d in "$data/"{train,dev,test}; do
    if ! [ -f "$d/ref.trn_phone" ]; then
        echo "Writing trn files in '$d'"
        ./step.py metadata-to-trn "$d/metadata.csv" "$d/ref.trn"
        filter "$d/ref.trn"
        if $only; then exit 0; fi
    fi
    if [ $bootstrap_samples -gt 0 ] && ! [ -f "$d/utt2rec" ]; then
        echo "Writing $d/utt2rec"
        sed 's/.*(\(.*\))$/\1/; s/\(.*\)_\(.*\)$/\1_\2 \2/' "$d/ref.trn" \
            > "$d/utt2rec"
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
        ./prep/ngram_lm.py -o $lm_ord -t 0 1 < "etc/lm_text.txt" > "${lm}_"
        mv "${lm}"{_,}
        if $only; then exit 0; fi
    fi
fi

for part in dev test; do
    if ! [ -f "$exp/decode/${part}_${name}.trn_phone" ]; then
        echo "Decoding $part"
        ./prep/logits-to-trn-via-pyctcdecode.py \
            --char "${lm_args[@]}" \
            --words "etc/lm_words.txt" \
            --width $width \
            --beta $beta \
            --alpha-inv $alpha_inv \
            --token2id "$exp/token2id" \
            "$exp/decode/"{logits/$part,${part}_$name.trn}
        filter "$exp/decode/${part}_$name.trn"
        if $only; then exit 0; fi
    fi
done

for er in filt char phone; do
    echo "===================================================================="
    echo "                       ERROR TYPE: $er                              "
    echo "===================================================================="
    echo ""
    for part in dev test; do
        ./prep/error-rates-from-trn.py \
            --suppress-warning --ignore-empty-refs --differences \
            --bootstrap-samples "$bootstrap_samples" \
            --bootstrap-utt2grp "$data/$part/utt2rec" \
            "$data/$part/ref.trn_$er" \
            "$exp/decode/${part}_"*".trn_$er"
        echo ""
    done
done
