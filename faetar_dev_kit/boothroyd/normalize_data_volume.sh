#! /usr/bin/env bash

# Copyright 2023 Sean Robertson
# Apache 2.0

echo "$0 $*"

pref="0.00001"
l0=70
samp_max=32767
cleanup=true
nj=10
cmd=run.pl

. ./path.sh
. utils/parse_options.sh

if [ $# -ne 2 ]; then
  echo "Usage: $0 [opts] <src-data> <dest-data>"
  echo "e.g. $0 data/dev_clean data/dev_clean_norm"
  echo ""
  echo "Options:"
  echo "--pref FLOAT    The reference amplitude. Default 0.00001"
  echo "--l0 FLOAT      The reference level (dB). Defaults to 70"
  echo "--samp-max NAT  The maximum integer value a sample can take. Used "
  echo "                to scale values between [-1, 1]. Defaults to 32767 "
  echo "                (pcm16 max)"
  echo "--cleanup (true|false)"
  echo "                Whether to delete temporary files when done. Default "
  echo "                true"
  exit 1
fi

src="$1"
dst="$2"
tmpdir="$dst/tmp"
logdir="$dst/log/normalize_data_volume"
tsrc="$tmpdir/tsrc"

for x in "$src/wav.scp"; do
  if [ ! -f "$x" ]; then
    echo "$0: file '$x' does not exist!"
    exit 1
  fi
done

set -e

mkdir -p "$tsrc" "$logdir" "$dst"

./utils/split_data.sh --per-utt "$src" "$nj"

$cmd JOB=1:$nj $logdir/get_wav_dc.JOB.log \
  ./local/get_wav_dc.py --inv-scale-by "$samp_max" \
  "scp,s,o:$src/split${nj}utt/JOB/wav.scp" "ark,t:$tmpdir/dc.JOB.txt"

for (( n=1; n <= nj; n+= 1 )); do
  awk '{print "sox - -t wav - dcshift",-$2,"|"}' "$tmpdir/dc.$n.txt" |
    paste -d ' ' $src/split${nj}utt/$n/wav.scp - > "$tmpdir/wav.offset.$n.scp" &
done
wait

$cmd JOB=1:$nj $logdir/get_wav_power.JOB.log \
  ./local/get_wav_power.py \
    --magnitude=true --normalize=true --inv-scale-by "$samp_max" \
    scp,s,o:$tmpdir/wav.offset.JOB.scp ark,t:$tmpdir/mag.JOB.txt

# copy the data directory over to tsrc. This makes it easier to copy
# only the relevant files to dir
./utils/copy_data_dir.sh "$src" "$tsrc"
rm -f "$tsrc/"{feats.scp,cmvn.scp}


for (( n=1; n <= nj; n+= 1 )); do
  paste -d ' ' $tmpdir/{dc,mag}.$n.txt |
    awk -v "pref=$pref" -v "l0=$l0" '
BEGIN {coeff=pref * exp(log(10) * l0 / 20)}
{print "sox - -t wav - dcshift",-$2,"vol",coeff / $4,"|"}' |
    paste -d ' ' $src/split${nj}utt/$n/wav.scp -
done | sort -k 1,1 -u > "$tsrc/wav.scp"

if $validate; then
  ./utils/validate_data_dir.sh --no-feats "$tsrc"
  
  ./local/get_wav_power.py \
      --normalize=true --inv-scale-by \
      "$samp_max" "scp,s,o:$tsrc/wav.scp" "ark:-" 2> /dev/null |
    python -c "l0=$l0;pref=$pref;"'
from pydrobert.kaldi.io import open
import numpy as np
exp = 10 ** (l0 / 10) * (pref ** 2)
f = open("ark:-", "b")
for utt, act in f.items():
  assert np.isclose(exp, act, rtol=1e-3), f"{utt}: exp={exp}, act={act}"
'
fi

cp "$tsrc"/* "$dst"

! $cleanup || rm -rf "$tmpdir"