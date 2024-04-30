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


exp=exp/mms-10min
train_part=10min
asr_config=conf/ml_superb/tuning/train_asr_fbank_single.yaml
inference_config=conf/ml_superb/decode_asr.yaml

if [ "$train_part" != "10min" ] && [ "$train_part" != "1h" ]; then
    echo "'$train_part' is not 10min or 1h! set -p appropriately!"
    exit 1
fi
for d in "data/"{$train_part,dev,test}; do
    if ! [ -d "$d" ]; then
        echo -e "'$d' is not a directory! ESPNet requires data in data/ folder!"
        exit 1
    fi
done
if ! mkdir -p "$exp" 2> /dev/null; then
    echo -e "Could not create '$exp'! set -e appropriately!"
    exit 1
fi

set -eo pipefail

if ! [ -f "espnet/egs2/ml_superb/asr1/asr.sh" ]; then
    echo "Initializing Git submodule"
    git submodule update --init --remote espnet
    if $only; then exit 0; fi
fi

for d in "data/"{$train_part,dev,test}; do
    if ! [ -f "$d/uttlist" ]; then
        echo "Constructing '$d/uttlist'"
        join \
            <(find "$d/" -maxdepth 1 -name '*.wav' -exec basename {} \; | cut -d '.' -f 1 | sort) \
            <(find "$d/" -maxdepth 1 -name '*.txt' -exec basename {} \; | cut -d '.' -f 1 | sort) \
            > "$d/uttlist_"
        mv "$d/uttlist"{_,}
        if $only; then exit 0; fi
    fi
    if ! [ -f "$d/wav.scp" ]; then
        echo "Constructing '$d/wav.scp'"
        awk -v "d=$d" '{print $1" "d"/"$1".wav"}' "$d/uttlist" > "$d/wav.scp_"
        mv "$d/wav.scp"{_,}
        if $only; then exit 0; fi
    fi
    if ! [ -f "$d/text" ]; then
        echo "Constructing '$d/text'"
        awk -v "d=$d" '{utt=$1; fn=d"/"utt".txt"; getline < fn; print utt" "$0}' "$d/uttlist" > "$d/text_"
        mv "$d/text"{_,}
        if $only; then exit 0; fi
    fi
    if ! [ -f "$d/utt2spk" ]; then
        echo "Constructing '$d/utt2spk'"
        awk -F '_' '{print $1" "$0}' "$d/uttlist" > "$d/utt2spk_"
        mv "$d/utt2spk"{_,}
        if $only; then exit 0; fi
    fi
done


for stage in $(seq 2 32); do
    if ! [ -f "$exp/.stage.$stage.done" ]; then
        echo "Running ESPNet stage $stage"
        ./espnet/egs2/TEMPLATE/asr1/asr.sh \
            --ngpu 1 \
            --stage $stage --stop_stage $stage \
            --nj 32 --inference-nj 32 \
            --gpu_inference false \
            --lang fae \
            --inference_asr_model "valid.loss.ave.pth" \
            --local_data_opts "--duration $train_part --lid false --multilingual false --single-lange fae" \
            --use_lm false \
            --token_type char \
            --feats_type raw \
            --feats_normalize utterance_mvn \
            --asr_config "${asr_config}" \
            --train_set "$train_part" \
            --valid_set "dev" \
            --test_sets "$train_part dev test" \
            --asr_tag "$(basename "${asr_config}" .yaml)_fae_$train_part" \
            --expdir "$exp" \
            --dumpdir "$exp/dump" \
            --asr_stats_dir "$exp/asr_stats_fae_$train_part"
        touch "$exp/.stage.$stage.done"
        if $only; then exit 0; fi
    fi
done
