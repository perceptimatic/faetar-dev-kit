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

usage="Usage: $0 [-h] [-o] [-g] [-e DIR] [-d DIR] [-p {10min,1h}] [-c FILE] [-C FILE]"
only=false
gpu_inference=false
train_jobs=32
inference_jobs=32
data=data
dump=dump
exp=exp
train_part=10min
asr_config=conf/ml_superb/tuning/train_asr_mms_single.yaml
inference_config=conf/ml_superb/decode_asr.yaml
help="Train and decode with an ML-SUPERB baseline

Options
    -h          Display this help message and exit
    -o          Run only the next step of the script
    -g          Enable GPU inference
    -e DIR      The experiment directory (default: '$exp')
    -d DIR      The data directory (default: '$data')
    -d DIR      The data dump directory (default: '$dump')
    -p {10min,1h}
                The amount of data to train with (default: '$train_part')
    -c FILE     Path to asr config YAML (default: '$asr_config')
    -C FILE     Path to inference config YAML (default: '$inference_config')
    -j NAT      Number of training jobs (default: $train_jobs)
    -J NAT      Number of inference jobs (default: $inference_jobs)"

while getopts "hoge:d:D:p:c:C:j:J:" name; do
    case $name in
        h)
            echo "$usage"
            echo ""
            echo "$help"
            exit 0;;
        o)
            only=true;;
        g)
            gpu_inference=true;;
        e)
            exp="$OPTARG";;
        d)
            data="$OPTARG";;
        D)
            dump="$OPTARG";;
        p)
            train_part="$OPTARG";;
        c)
            asr_config="$OPTARG";;
        C)
            inference_config="$OPTARG";;
        j)
            train_jobs="$OPTARG";;
        J)
            inference_jobs="$OPTARG";;
        *)
            echo -e "$usage"
            exit 1;;
    esac
done
shift $(($OPTIND - 1))
if [ "$train_part" != "10min" ] && [ "$train_part" != "1h" ]; then
    echo "'$train_part' is not 10min or 1h! set -p appropriately!"
    exit 1
fi
for d in "$data/"{$train_part,dev,test}; do
    if ! [ -d "$d" ]; then
        echo -e "'$d' is not a directory! set -d appropriately"
        exit 1
    fi
done
if ! mkdir -p "$exp" 2> /dev/null; then
    echo -e "Could not create '$exp'! set -e appropriately!"
    exit 1
fi
if ! mkdir -p "$dump" 2> /dev/null; then
    echo -e "Could not create '$dump'! set -D appropriately!"
    exit 1
fi
if ! [ -f "$asr_config" ]; then
    echo -e "'$asr_config' is not a file! Set -c appropriately!"
    exit 1
fi
if ! [ -f "$inference_config" ]; then
    echo -e "'$inference_config' is not a file! Set -C appropriately!"
    exit 1
fi
if ! [ "$train_jobs" -gt 0 ] 2> /dev/null; then
    echo -e "$train_jobs is not a natural number! set -j appropriately!"
    exit 1
fi
if ! [ "$inference_jobs" -ge 0 ] 2> /dev/null; then
    echo -e "$inference_jobs is not a non-negative int! set -J appropriately!"
    exit 1
fi


set -eo pipefail

if ! [ -f "espnet/egs2/ml_superb/asr1/asr.sh" ]; then
    echo "Initializing Git submodule"
    git submodule update --init --remote espnet
    if $only; then exit 0; fi
fi

data="$(cd "$data"; pwd -P)"
exp="$(cd "$exp"; pwd -P)"
dump="$(cd "$dump"; pwd -P)"
asr_config="$(cd "$(dirname "$asr_config")"; pwd -P)/$(basename "$asr_config")"
inference_config="$(cd "$(dirname "$inference_config")"; pwd -P)/$(basename "$inference_config")"
asr_tag="$(basename "${asr_config}" .yaml)_fae_$train_part"

pushd espnet

if ! [ -f tools/venv/bin/python3 ]; then
    echo "Creating espnet venv"
    pushd tools
    ./setup_venv.sh "$(command -v python3)"
    if $only; then exit 0; fi
    popd
fi

if ! [ -f tools/venv/.make.done ]; then
    echo "Making espnet dependencies"
    pushd tools
    make s3prl.done
    touch venv/.make.done
    if $only; then exit 0; fi
    popd
fi

pushd egs2/ml_superb/asr1

if ! [ -f downloads/hlvc/.done ]; then
    echo "Formatting data for ML-SUPERB in $PWD/downloads"
    mkdir -p downloads/hlvc/fae/wav
    find "$data/1hr" "$data/dev" "$data/test" -name '*.wav' -exec ln -sf {} downloads/hlvc/fae/wav/ \;
    find "$data/1hr" -name '*.txt' |
        sort |
        awk -F "/" '{split($NF, bn, "."); getline < $0; print bn[1]" A "$0}' > downloads/hlvc/fae/transcript_1h_train.txt
    find "$data/10min" -name '*.txt' |
        sort |
        awk -F "/" '{split($NF, bn, "."); getline < $0; print bn[1]" A "$0}' > downloads/hlvc/fae/transcript_10min_train.txt
    find "$data/dev" -name '*.txt' |
        sort |
        awk -F "/" '{split($NF, bn, "."); getline < $0; print bn[1]" A "$0}' > downloads/hlvc/fae/transcript_10min_dev.txt
    cp -f downloads/hlvc/fae/transcript_{10min,1h}_dev.txt
    find "$data/test" -name '*.txt' |
        sort |
        awk -F "/" '{split($NF, bn, "."); getline < $0; print bn[1]" A "$0}' > downloads/hlvc/fae/transcript_10min_test.txt
    cp -f downloads/hlvc/fae/transcript_{10min,1h}_test.txt
    touch downloads/hlvc/.done
    if $only; then exit 0; fi
fi

for stage in $(seq 1 12); do
    if ! [ -f "$exp/.stage.$stage.done" ]; then
        echo "Running ML-SUPERB stage $stage"
        ./asr.sh \
            --ngpu 1 \
            --stage $stage --stop_stage $stage \
            --nj $train_jobs --inference-nj $inference_jobs \
            --gpu_inference $gpu_inference \
            --lang fae \
            --inference_asr_model "valid.loss.ave.pth" \
            --local_data_opts "--duration $train_part --lid false --multilingual false --single-lang fae" \
            --use_lm false \
            --token_type char \
            --feats_type raw \
            --feats_normalize utterance_mvn \
            --asr_config "${asr_config}" \
            --train_set "train_${train_part}_fae" \
            --valid_set "dev_${train_part}_fae" \
            --test_sets "dev_${train_part}_fae test_${train_part}_fae" \
            --asr_tag "$asr_tag" \
            --expdir "$exp" \
            --dumpdir "$dump" \
            --asr_stats_dir "$exp/asr_stats_fae_$train_part"
        touch "$exp/.stage.$stage.done"
        if $only; then exit 0; fi
    fi
done

popd

for part in dev test; do
    if [ ! -f "$exp/decode/${part}_beam1.trn" ]; then
        echo "Writing $exp/decode/${part}_beam1.trn"
        mkdir -p "$exp/decode"
        src="$exp/asr_$asr_tag/inference_asr_model_valid.loss.ave/org/${part}_${train_part}_fae/text"
        if ! [ -f "$src" ]; then
            src="$exp/asr_$asr_tag/inference_asr_model_valid.loss.ave/${part}_${train_part}_fae/text"
        fi
        cat "$src" |
            sort |
            awk -F ' ' '{utt=$1; for (i = 1; i < NF; i++) $i = $(i + 1); $NF = "("utt")"; print}' \
            > "$exp/decode/${part}_beam1.trn_"
        mv "$exp/decode/${part}_beam1.trn"{_,}
        if $only; then exit 0; fi
    fi
done
