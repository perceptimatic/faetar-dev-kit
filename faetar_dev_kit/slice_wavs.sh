#! /usr/bin/env bash

# Copyright 2024 Michael Ong
# Apache 2.0

# splits .wav files starting fom the centre of each file until the resulting files are shorter than the threshhold

echo "$0 $*"

usage="Usage: $0 [-h] [-d DIR] [-s POSREAL] [-o DIR]"
data_dir=
split_thresh=20
out_dir=
help="Splits .wav files that have a duration longer than the threshhold + copies all other .wav files

Options
    -h          Display this help message and exit
    -d DIR      The data directory (default: '$data_dir')
    -s POSREAL  The threshhold (in seconds) for splitting (default: '$split_thresh')
    -o DIR      The output directory (default: '$out_dir')"

while getopts "hd:s:o:" name; do
    case $name in
        h)
            echo "$usage"
            echo ""
            echo "$help"
            exit 0;;
        d)
            data_dir="$OPTARG";;
        s)
            split_thresh="$OPTARG";;
        o)
            out_dir="$OPTARG";;
        *)
            echo -e "$usage"
            exit 1;;
    esac
done
shift $(($OPTIND - 1))
if [ ! -d "$data_dir" ]; then
    echo -e "'$data_dir' is not a directory! Set -d appropriately!"
    exit 1
fi
if ! [[ "$split_thresh" =~ ^[0-9]+\.?[0-9]*$ ]] 2> /dev/null; then
    echo -e "$split_thresh is not a positive real number! set -p appropriately, or add a leading zero!"
    exit 1
fi
if ! mkdir -p "$out_dir" 2> /dev/null; then
    echo -e "Could not create '$out_dir'! set -o appropriately!"
    exit 1
fi

set -e

mkdir -p "$out_dir"

for file in "$data_dir"/*.wav; do
    filename="$(basename "$file" .wav)"
    dur="$(soxi -D "$file")"
    if [ "$(bc -l <<< ""$dur" > $split_thresh")" -eq 1 ]; then
        split_dur="$dur"
        while [ "$(bc -l <<< ""$split_dur" > $split_thresh")" -eq 1 ]; do
            split_dur="$(bc -l <<< "$split_dur / 2")"
        done
        sox "$file" "$out_dir/%n_$filename.wav" trim 0 "$split_dur" : newfile : restart
    else
        cp "$file" "$out_dir"
    fi
done


for file in data/mms_lsah_CV/*/*.txt; do
    awk \
        '{
            gsub(/d[zʒ]ː|t[sʃ]ː|d[zʒ]|t[sʃ]|[bdfhjklmnprstvwzŋɡɣɲʃʎʒ]ː|[bdfhjklmnprstvwzŋɡɣɲʃʎʒ]/, "C");
            gsub(/[aeiouɔəɛɪʊʌ]ː|[aeiouɔəɛɪʊʌ]/, "V");
            print $0;
        }' "$file" > "$file"_
    mv "$file"{_,}
done