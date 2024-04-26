#!/usr/bin/env bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 bench_corpus_dir"
    exit 1
fi

set -eo pipefail

# splits text into individual phones
function split_text () {
  text="$1"

  awk \
  'BEGIN {
    FS = " ";
    OFS = " ";
  }

  {
    gsub(/\[fp\]|d[zʒ]ː|tʃː|d[zʒ]|tʃ|\Sː|\S/, "& ");
    gsub(/ +/, " ");
    print $0;
  }' "$text"
}

bench_dir="$1"
conf_dir="conf_lstm"
out_dir="data_lstm"
partitions=(train test dev)

mkdir -p "$out_dir"/prep/
mkdir -p "$conf_dir"

if [ ! -f "$conf_dir/model.yaml" ]; then
  python3 prep/asr-baseline.py --print-model-yaml "$conf_dir/model.yaml"
fi
if [ ! -f "$conf_dir/data.yaml" ]; then
  python3 prep/asr-baseline.py train --print-data-yaml "$conf_dir/data.yaml"
fi
if [ ! -f "$conf_dir/training.yaml" ]; then
  python3 prep/asr-baseline.py train --print-training-yaml "$conf_dir/training.yaml"
fi

for partition in "${partitions[@]}"; do
    :> "$out_dir"/prep/"wav_${partition}.scp"
    :> "$out_dir"/prep/"trn_$partition"
    for file in "$bench_dir"/"$partition"/*.wav; do
        filename="$(basename "$file" .wav)"
        printf "%s %s\n" "$filename" "$file" >> "$out_dir"/prep/"wav_${partition}.scp"
        printf "%s(%s)\n" "$(split_text "${file%%.wav}.txt")" "$filename" >> "$out_dir"/prep/"trn_$partition"
    done

    if [[ $partition == "train" ]]; then
        rev "$out_dir"/prep/trn_train |
        cut -d ' ' -f 2- |
        rev |
        tr ' ' $'\n' |
        sort -u |
        awk 'NF != 0' |
        awk '{print $0,NR-1}' > "$out_dir"/prep/"token2id"
    fi

    signals-to-torch-feat-dir \
    "$out_dir"/prep/"wav_${partition}.scp" prep/conf/feats/fbank_41.json "$out_dir"/"$partition"/feat/
    trn-to-torch-token-data-dir \
    "$out_dir"/prep/"trn_$partition" "$out_dir"/prep/token2id "$out_dir"/"$partition"/ref/
    num_filts="$(awk 'NR == 3 {print $2}' <<< "$(get-torch-spect-data-dir-info --strict "$out_dir"/"$partition")")"
    if [[ num_filts -ne 41 ]]; then
        echo "$partition has incorrect feature dims"
        exit 1
    fi
done


