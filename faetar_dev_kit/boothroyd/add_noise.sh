#! /usr/bin/env bash

# Copyright 2023 Sean Robertson
# Apache 2.0

echo "$0 $*"

cleanup=true
cmd=run.pl
noise_type=whitenoise
samp_max=32767
nj=10
validate=false

. ./path.sh
. utils/parse_options.sh

if [ $# -ne 3 ] && [ $# -ne 4 ]; then
  echo "Usage: $0 [opts] <src-data> <snr> <dest-data> [<noise-dir>]"
  echo "e.g. $0 data/dev_clean_norm 0 data/dev_clean_snr_0 mfcc"
  echo ""
  echo "snr is in decibels"
  echo ""
  echo "Options:"
  echo "--noise-type TYPE  The type of noise generated with the sox synth"
  echo "                   command. Defaults to whitenoise"
  echo "--samp-max NAT  The maximum integer value a sample can take. Used "
  echo "                to scale values between [-1, 1]. Defaults to 32767 "
  echo "                (pcm16 max)"
  echo "--validate (true|false)"
  echo "                If set, will double-check the resulting wavs' power "
  echo "                matches reference levels. Default false"
  echo "--cleanup (true|false)"
  echo "                Whether to delete temporary files when done. Default "
  echo "                true"
  exit 1
fi

src="$1"
snr="$2"
dst="$3"
ndir="${4:-"$3/noise"}"
tmpdir="$dst/tmp"
logdir="$dst/log/add_noise_$snr"
tsrc="$tmpdir/tsrc"

for x in "$src/wav.scp"; do
  if [ ! -f "$x" ]; then
    echo "$0: file '$x' does not exist!"
    exit 1
  fi
done

set -eo pipefail

mkdir -p "$tsrc" "$dst" "$ndir"

./utils/data/get_reco2dur.sh "$src"
max_dur=$(cut -d ' ' -f 2 "$src/reco2dur" | awk -v m=0 '{if ($1 > m) m = $1} END {print m}')

nfile="$ndir/$noise_type.$max_dur.wav"
# -R flag should keep this file the same, no matter how many times it's
# called
sox -R -b 16 -r 16k -n $nfile synth $max_dur $noise_type vol 0.99

./utils/split_data.sh --per-utt "$src" "$nj"

for (( n=1; n <= nj; n+=1 )); do
  ./utils/filter_scp.pl "$src/split${nj}utt/$n/wav.scp" "$src/reco2dur" |
    awk -v "nfile=$nfile" '{print $1,"sox",nfile, "-t wav - trim 0",$2,"|"}' \
      > "$tmpdir/noise.wav.$n.scp" &
done
wait

$cmd JOB=1:$nj $logdir/get_noise_power.JOB.log \
  ./local/get_wav_power.py \
    --magnitude=true --normalize=true --inv-scale-by "$samp_max" \
    scp,s,o:$tmpdir/noise.wav.JOB.scp ark,t:$tmpdir/noise_mag.JOB.txt

$cmd JOB=1:$nj $logdir/get_wav_power.JOB.log \
  ./local/get_wav_power.py \
    --magnitude=true  --normalize=true --inv-scale-by "$samp_max" \
    scp,s,o:$src/split${nj}utt/JOB/wav.scp ark,t:$tmpdir/sig_mag.JOB.txt

for (( n=1; n <= nj; n+= 1)); do
  # vol=sqrt(sigpow/(10 ** (SNR / 10) * noisepow))
  paste -d ' ' "$tmpdir/"{sig,noise}_mag.$n.txt |
    awk -v "snr=$snr" '
BEGIN {snrmag=exp(log(10) * (snr / 20))}
{print $1,$2 / ($4 * snrmag)}
' > "$tmpdir/noise_vol.$n.txt" &
done
wait

if $validate; then
  for (( n=1; n <= nj; n+=1 )); do
    paste -d ' ' "$tmpdir/noise.wav.$n.scp" "$tmpdir/noise_vol.$n.txt" |
      awk '{vol=$NF; NF -=2; print $0,"sox -v",vol,"- -t wav - |"}' |
      ./local/get_wav_power.py \
        --normalize=true --inv-scale-by "$samp_max" "scp:-" "ark:-" 2> /dev/null |
    python -c "snr=$snr; sig_pow='ark,s:$tmpdir/sig_mag.$n.txt';"'
from pydrobert.kaldi.io import open
import numpy as np
utt2sigmag = open(sig_pow, "b", "r+")
f = open("ark:-", "b")
for utt, noisepow in f.items():
  sigpow = utt2sigmag[utt] ** 2
  act = 10 * (np.log10(sigpow) - np.log10(noisepow))
  assert np.isclose(act, snr, rtol=1e-3), (
    f"{utt}: exp={snr}, act={act}, sigpow={sigpow}, noisepow={noisepow}"
  )
'
  done
fi

echo "Copying"

./utils/copy_data_dir.sh "$src" "$tsrc"
rm -f "$tsrc/"{feats.scp,cmvn.scp}

for (( n=1; n <= nj; n+=1 )); do
  ./utils/filter_scp.pl "$src/split${nj}utt/$n/wav.scp" "$src/reco2dur" |
    paste -d ' ' "$tmpdir/noise_vol.$n.txt" - |
    awk -v nfile=$nfile \
      '{print "sox -m - -v",$2,nfile,"-t wav - trim 0",$4,"|"}' |
    paste -d ' ' "$src/split${nj}utt/$n/wav.scp" -
done | sort -k 1,1 -u > "$tsrc/wav.scp"

if $validate; then
  ./utils/validate_data_dir.sh --no-feats "$tsrc"
fi

cp "$tsrc"/* "$dst"

! $cleanup || rm -rf "$tmpdir"