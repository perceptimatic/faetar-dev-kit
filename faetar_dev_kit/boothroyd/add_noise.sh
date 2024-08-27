#! /usr/bin/env bash

# Copyright 2024 Sean Robertson, Michael Ong
# Apache 2.0

echo "$0 $*"

usage="Usage: $0 [-h] [-n TYPE] [-b NAT] [-d DIR] [-s REAL] [-o DIR] [-p DIR]"
noise_type=whitenoise
bit_depth=16
data_dir=
snr=
out_dir=
partition=
help="Adds generated noise to .wav files

Options
    -h          Display this help message and exit
    -n TYPE     The type of noise generated with the sox synth
                command (default: '$noise_type')
    -b NAT      The bit depth of the output (default: '$bit_depth')
    -d DIR      The data directory (default: '$data_dir')
    -s REAL     The signal to noise ratio in dB (default: '$snr')
    -o DIR      The output directory (default: '$out_dir')
    -p DIR      The partition subdirectory (default: '$out_dir/$partition')"

while getopts "hn:b:d:s:o:p:" name; do
    case $name in
        h)
            echo "$usage"
            echo ""
            echo "$help"
            exit 0;;
        n)
            noise_type="$OPTARG";;
        b)
            bit_depth="$OPTARG";;
        d)
            data_dir="$OPTARG";;
        s)
            snr="$OPTARG";;
        o)
            out_dir="$OPTARG";;
        p)
            partition="$OPTARG";;
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
if ! mkdir -p "$out_dir" 2> /dev/null; then
    echo -e "Could not create '$out_dir'! set -o appropriately!"
    exit 1
fi
if ! mkdir -p "$out_dir/$partition" 2> /dev/null; then
    echo -e "Could not create '$$out_dir/$partition'! set -p appropriately!"
    exit 1
fi
if ! [ "$(grep -Ew "sine|square|triangle|sawtooth|trapezium|exp|(|white|pink|brown)noise" <<< "$noise_type")" ] 2> /dev/null; then
    echo -e "$noise_type is not a valid noise type! set -n appropriately!"
    exit 1
fi
if ! [ "$bit_depth" -gt 0 ] 2> /dev/null; then
    echo -e "$bit_depth is not a natural number! set -b appropriately!"
    exit 1
fi
if ! [[ "$snr" =~ ^-?[0-9]+\.?[0-9]*$ ]] 2> /dev/null; then
    echo -e "$snr is not a real number! set -s appropriately, or add a leading zero!"
    exit 1
fi

set -eo pipefail

mkdir -p "$out_dir/noise"
mkdir -p "$out_dir/$partition/snr${snr}"

max_dur=0

for file in "$data_dir"/*.wav; do
  file_dur="$(soxi -D "$file")"
  max_dur="$(bc -l <<< "if ($file_dur > $max_dur) {$file_dur;} else {$max_dur;}")"
done

full_noise_file="$out_dir/noise/${noise_type}_${partition}_${max_dur}.wav"

# -R flag should keep this file the same, no matter how many times it's
# called
sox -R -b 16 -r 16k -n $full_noise_file synth $max_dur $noise_type

for file in "$data_dir"/*.wav; do
  filename="$(basename "$file")"
  file_dur="$(soxi -D "$file")"
  trimmed_noise_rms_amp="$(sox "$full_noise_file" -b "$bit_depth" -n trim 0 "$file_dur" stat 2>&1 | awk '/RMS\s+amplitude:/ {print $3}')"
  file_rms_amp="$(sox "$file" -n stat 2>&1 | awk '/RMS\s+amplitude:/ {print $3}')"
  # calculates $file_rms_amp / ((10^($snr / 20)))
  target_rms_amp="$(bc -l <<< "$file_rms_amp / e(l(10) * ($snr / 20))")"
  vol_shift="$(bc -l <<< "$target_rms_amp / $trimmed_noise_rms_amp")"
  out_path="$out_dir/$partition/snr${snr}/$filename"

  sox -m "$file" -v "$vol_shift" "$full_noise_file" -t wav "$out_path" trim 0 "$file_dur"
done