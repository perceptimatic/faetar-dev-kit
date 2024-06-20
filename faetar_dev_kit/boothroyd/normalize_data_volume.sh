#! /usr/bin/env bash

# Copyright 2024 Sean Robertson, Michael Ong
# Apache 2.0

echo "$0 $*"

usage="Usage: $0 [-h] [-p NINT] [-l NAT] [-b NAT] [-d DIR] [-o DIR]"
pref="-5"
l0=70
bit_depth=16
data_dir=
out_dir=
help="Normalize volume of .wav files to a given reference amplitude + dB level

Options
    -h          Display this help message and exit
    -p NINT     The reference amplitude log 10 (default: '$pref')
    -l NAT      The reference level (dB) (default: '$l0')
    -b NAT      The bit depth of the output (default: '$bit_depth')
    -d DIR      The data directory (default: '$data_dir')
    -o DIR      The output directory (default: '$out_dir')"

while getopts "hp:l:b:d:o:" name; do
    case $name in
        h)
            echo "$usage"
            echo ""
            echo "$help"
            exit 0;;
        p)
            pref="$OPTARG";;
        l)
            l0="$OPTARG";;
        b)
            bit_depth="$OPTARG";;
        d)
            data_dir="$OPTARG";;
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
if ! mkdir -p "$out_dir" 2> /dev/null; then
    echo -e "Could not create '$out_dir'! set -o appropriately!"
    exit 1
fi
if ! [ "$pref" -lt 0 ] 2> /dev/null; then
    echo -e "$pref is not a negative integer! set -p appropriately!"
    exit 1
fi
if ! [ "$l0" -gt 0 ] 2> /dev/null; then
    echo -e "$l0 is not a natural number! set -l appropriately!"
    exit 1
fi
if ! [ "$bit_depth" -gt 0 ] 2> /dev/null; then
    echo -e "$bit_depth is not a natural number! set -b appropriately!"
    exit 1
fi

set -e

mkdir -p "$out_dir"

for file in "$data_dir"/*.wav; do
  filename="$(basename "$file")"
  dc_shift="$(sox "$file" -n stat 2>&1 | awk '/Mean\s+amplitude:/ {print -$3}')"
  file_rms_amp="$(sox "$file" -n dcshift "$dc_shift" stat 2>&1 | awk '/RMS\s+amplitude:/ {print $3}')"
  # calculates 10^$pref * 10^($l0/20)
  target_rms_amp="$(bc -l <<< "e(l(10) * ($pref + ($l0 / 20)))")"
  vol_shift="$(bc -l <<< "$target_rms_amp / $file_rms_amp")"
  # sets mean amplitude to 0 
  # and sets rms amplitude to the value corresponding to l0 dB above the reference amplitude 10^$pref
  # by default this causes the output rms amplitude to be 0.0316227766 / 10^(-3/2)
  sox "$file" -b "$bit_depth" "$out_dir"/"$filename" dcshift "$dc_shift" vol "$vol_shift"
done