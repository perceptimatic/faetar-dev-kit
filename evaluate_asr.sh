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

usage="Usage: $0 [-h] [-o] [-e DIR] [-d DIR] [-r {wer|cer|per}] [-n NNINT]"
only=false
data=data
model=
part=test
exp=exp
er=per
bootstrap_samples=0
help="Run ASR evaluation on a partition

Options
    -h          Display this help message and exit
    -o          Run only the next step of the script
    -e DIR      The experiment directory (default: '$exp')
    -m {lstm|mms_lsah|ml_superb}
                The model name
    -d DIR      The data directory (default: '$data')
    -p DIR      The partition directory (default: '$part')
    -r {wer|cer|per}
                The type of error rate to compute (default: '$er')
    -n NNINT    Bootstrap samples. 0 is no bootstrap (default: $bootstrap_samples)"

while getopts "hoe:m:d:p:r:n:" name; do
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
        m)
            exp="$OPTARG";;
        d)
            data="$OPTARG";;
        p)
            part="$OPTARG";;
        r)
            er="$OPTARG";;
        n)
            bootstrap_samples="$OPTARG";;
        *)
            echo -e "$usage"
            exit 1;;
    esac
done
shift $(($OPTIND - 1))
if [ ! -d "$exp" ]; then
    echo -e "'$exp' is not a directory! set -e appropriately!"
    exit 1
fi
if [ ! -d "$data" ]; then
    echo -e "'$data' is not a directory! set -d appropriately!"
    exit 1
fi
if [ ! -d "$part" ]; then
    echo -e "'$part' is not a directory! set -p appropriately!"
    exit 1
fi
if [ "$er" != "wer" ] && [ "$er" != "cer" ] && [ "$er" != "per" ]; then
    echo "'$er' is not one of wer, cer, or per! set -r appropriately!"
    exit 1
fi
if [ "$model" != "lstm" ] && [ "$model" != "mms_lsah" ] && [ "$model" != "ml_superb" ]; then
    echo "'$model' is not one of lstm, mms_lsah, or ml_superb! set -m appropriately!"
    exit 1
fi
if ! [ "$bootstrap_samples" -ge 0 ] 2> /dev/null; then
    echo -e "$bootstrap_samples is not a non-negative int! set -n appropriately!"
    exit 1
fi

hyps=( $(find "$exp/" -name "${part}_*.trn") )
if [ "${#hyps[@]}" = 0 ]; then
    echo -e "'$exp' contains no trn files! Set -e appropriately!"
    exit 1
fi

set -eo pipefail


filter() {
    fn="$1"
    sed 's/\[[^]][^]]*\] //g; s/<[^>][^>]*> //g;' "$fn" > "${fn}_wer_"
    mv "${fn}_wer"{_,}
    ./prep/word2subword.py "$fn"{_wer,_cer_}
    mv "${fn}_cer"{_,}
    sed 's/_ //g' "${fn}_cer" > "${fn}_per_"
    mv "${fn}_per"{_,}
}

if [ ! -f "prep/ngram_lm.py" ]; then
    echo "Initializing Git submodule"
    git submodule update --init --remote prep
    if $only; then exit 0; fi
fi

if ! [ -f "$data/$model/$part/ref.trn" ]; then
    echo "Writing '$data/$model/$part/ref.trn'"
    find "$data/$model/$part/" -maxdepth 1 -name '*.txt' | 
        sort |
        awk -v "d=$data/$model/$part" -F '/' '
{
    split($NF, bn, ".");
    getline < $0;
    print $0" ("bn[1]")";
}
END {
    if (NR == 0) {
        print "directory "d"contains no transcripts" > /dev/stderr;
        exit 1
    }
}' > "$data/$model/$part/ref.trn_"
    mv "$data/$model/$part/ref.trn"{_,}
    if $only; then exit 0; fi
fi

if ! [ -f "$data/$model/$part/utt2rec" ]; then
    echo "Writing $data/$model/$part/utt2rec"
    sed 's/.*(\(.*\))$/\1/; s/\(.*\)_\(.*\)$/\1_\2 \2/' "$data/$model/$part/ref.trn" \
        > "$data/$model/$part/utt2rec_"
    mv "$data/$model/$part/utt2rec"{_,}
    if $only; then exit 0; fi
fi

if ! [ -f "$data/$model/$part/ref.trn_$er" ]; then
    echo "Filtering '$data/$model/$part/ref.trn'"
    filter "$data/$model/$part/ref.trn"
    if $only; then exit 0; fi
fi

for i in "${!hyps[@]}"; do
    hyp="${hyps[i]}"
    if [ ! -f "${hyp}_${er}" ]; then
        echo "Filtering '$hyp'"
        filter "$hyp"
        if $only; then exit 0; fi
    fi
    hyps[$i]="${hyp}_${er}"
done

./prep/error-rates-from-trn.py \
    --suppress-warning --differences \
    --bootstrap-samples "$bootstrap_samples" \
    --bootstrap-utt2grp "$data/$model/$part/utt2rec" \
    "$data/$model/$part/ref.trn_${er}" "${hyps[@]}"
