#! /usr/bin/env bash

# Copyright 2023 Sean Robertson
# Apache 2.0

conf=conf      # configuration directory
exp=exp        # experiment directory
data=data      # data directory
latlm=tgsmall  # which lm to use to generate lattices
reslm=tgsmall  # which lm to use for lattice rescoring
mdl=chain_cleaned/tdnn_1d_sp      # which model to decode with
ivecmdl=nnet3_cleaned/extractor  # ivector extractor (tdnn mdl only)
part=dev_clean # partition to perform 
snr_low=-10      # lower bound (inclusive) of signal-to-noise ratio (SNR)
snr_high=30      # upper bound (inclusive) of signal-to-noise ratio (SNR)
pretrained_store=/ais/hal9000/sdrobert/librispeech_models  # where pretrained models are downloaded to
pretrained_url=https://kaldi-asr.org/models/13  # where to download pretrained models from
use_gpu=true

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

declare -A MNAME2PNAME=(
  ["rnnlm_lstm_1a"]="0013_librispeech_v1_lm"
  ["chain_cleaned/tdnn_1d_sp"]="0013_librispeech_v1_chain"
  ["nnet3_cleaned/extractor"]="0013_librispeech_v1_extractor"
)

set -e

mdldir="$exp/$mdl"
graphdir="$mdldir/graph_$latlm"

if [[ "$mdl" =~ tdnn ]]; then
  mfcc_suffix=_hires
  self_loop_args="--self-loop-scale 1.0"
else
  mfcc_suffix=
  ivecmdl=
  self_loop_args=
fi

for mname in "$mdl" "$reslm" ${ivecmdl:+"$ivecmdl"}; do
  pname="${MNAME2PNAME[$mname]}"
  if [ ! -z "$pname" ] && [ ! -f "$exp/$mname/.downloaded" ]; then
    echo "$0: $mname corresponds to pretrained model $pname. Downloading and extracting"
    mkdir -p "$pretrained_store" "$exp/$mname"
    ./local/download_pretrained_models.sh \
      "$pretrained_store" "$pretrained_url" "$pname"
    cp -r "$pretrained_store/exp/$mname/"* "$exp/$mname"
    touch "$exp/$mname/.downloaded"
  fi
done

if [[ ! "$mdl" =~ facebook ]]; then
  needed_files+=( $exp/$mdl/final.mdl $data/lang_test_$latlm/G.fst )
  if [ ! "$reslm" = "$latlm" ]; then
    if [[ "$reslm" =~ rnnlm ]]; then
      needed_files+=( "$exp/$reslm/final.raw" )
    else
      if [ ! -f "$data/lang_test_$reslm/G.fst" ]; then 
        needed_files+=( "$data/lang_test_$reslm/G.carpa" )
      fi
    fi
  fi
fi
for x in "${needed_files[@]}"; do
  if [ ! -f "$x" ]; then
    echo "$0: '$x' is not a file!"
    exit 1
  fi
done

if [[ "$mdl" =~ facebook ]]; then
  graphdir=data/lang
else
  if [ ! -f "$graphdir/HCLG.fst" ]; then
    utils/mkgraph.sh $self_loop_args \
      "$data/lang_test_$latlm" "$mdldir" "$graphdir"
  fi
fi

npart="${part}${mfcc_suffix}/norm"
parts=( $npart )

# Normalize data volume to same reference average RMS
if [ ! -f "$data/$npart/.complete" ]; then
  ./local/normalize_data_volume.sh "$data/$part" "$data/$npart"
  ./steps/make_mfcc.sh \
    --mfcc-config "$conf/mfcc${mfcc_suffix}.conf" --cmd "$train_cmd" --nj 40 \
    $data/$npart $exp/make_mfcc/$npart
  utils/fix_data_dir.sh $data/$npart
  steps/compute_cmvn_stats.sh $data/$npart $exp/make_mfcc/$npart
  touch "$data/$npart/.complete"
fi

for snr in $(seq $snr_low $snr_high); do
  spart="${part}${mfcc_suffix}/snr$snr"
  if [ ! -f "$data/$spart/.complete" ]; then
    # add noise at specific SNR, then compute feats + cmvn
    ./local/add_noise.sh $data/$npart $snr $data/$spart
    ./steps/make_mfcc.sh --mfcc-config "$conf/mfcc${mfcc_suffix}.conf" \
      --cmd "$train_cmd" --nj 40 \
      $data/$spart $exp/make_mfcc/$spart
    utils/fix_data_dir.sh $data/$spart
    steps/compute_cmvn_stats.sh $data/$spart $exp/make_mfcc/$spart
    touch "$data/$spart/.complete"
  fi
  parts+=( $spart )
done

for spart in "${parts[@]}"; do
  partdir="$data/$spart"
  if [[ "$mdl" =~ facebook ]]; then
    decodedir="$mdldir/decode_null_$spart"
    if [ ! -f "$decodedir/.complete" ]; then
      ./local/decode_transformer.sh "$graphdir" "$partdir" "$mdl" "$decodedir"
      touch "$decodedir/.complete"
    fi
  else  # not facebook
    latdecodedir="$mdldir/decode_${latlm}_$spart"
    if [[ "$reslm" = "$latlm" ]]; then
      decodedir="$latdecodedir"
    else
      decodedir="$mdldir/decode_${latlm}_rescore_${reslm}_$spart"
    fi

    # ivectors for tdnn
    if [[ "$mdl" =~ tdnn ]] && [ ! -f "$exp/$ivecmdl/ivectors_${spart}/.complete" ]; then
      steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 10 \
        "$data/$spart" "$exp/$ivecmdl" "$exp/$ivecmdl/ivectors_${spart}"
      touch "$exp/$ivecmdl/ivectors_${spart}/.complete"
    fi

    # decode the entire partition in the usual way using the lattice lm
    if [ ! -f "$latdecodedir/.complete" ]; then
      rm -rf "$latdecodedir"
      mkdir -p "$(dirname "$latdecodedir")"
      tmplatdecodedir="$exp/$mdl/tmp_decode"
      if [[ "$mdl" =~ tdnn ]]; then
        steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --use-gpu $use_gpu \
          --nj 10 --cmd "$decode_cmd" \
          --online-ivector-dir "$exp/$ivecmdl/ivectors_${spart}" \
          "$graphdir" "$partdir" "$tmplatdecodedir"
      else
        steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
          "$graphdir" "$partdir" "$tmplatdecodedir"
      fi
      mv "$tmplatdecodedir" "$latdecodedir"
      touch "$latdecodedir/.complete"
    fi

    # now rescore with the intended lm
    if [ ! -f "$decodedir/.complete" ]; then
      if [[ "$reslm" =~ rnnlm ]]; then
        # from libri_css/s5_css/run.sh
        rnnlm/lmrescore_pruned.sh \
          --cmd "$decode_cmd" \
          "$data/lang_test_$latlm" "$exp/$reslm" "$partdir" "$latdecodedir" \
          "$decodedir"
      else
        if [ -f "$data/lang_test_$reslm/G.fst" ]; then
            tmplatdecodedir="$exp/$mdl/tmp_decode"
            rm -rf "$tmplatdecodedir"
            cp -rf "$latdecodedir" "$tmplatdecodedir"
            steps/lmrescore.sh $self_loop_args --cmd "$decode_cmd" \
              "$data/"lang_test_{$latlm,$reslm} "$partdir" "$tmplatdecodedir" \
              "$decodedir"
            rm -rf "$tmplatdecodedir"
        else
          steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
            $data/lang_test_{$latlm,$reslm} "$partdir" "$latdecodedir" \
            "$decodedir"
        fi
      fi
      touch "$decodedir/.complete"
    fi
  fi

  if [ ! -f "$decodedir/wer_best" ]; then
    grep -H WER "$decodedir/"wer* | utils/best_wer.sh > "$decodedir/wer_best"
  fi

  if [ ! -f "$decodedir/uttwer_best" ]; then
    ./local/wer_per_utt.sh "$graphdir" "$decodedir/scoring"
    best_uttwer="$(awk '{gsub(/.*wer_/, "", $NF); gsub("_", ".", $NF);  print "scoring/"$NF".uttwer"}' "$decodedir/wer_best")"
    ln -sf "$best_uttwer" "$decodedir/uttwer_best"
    [ -f "$decodedir/uttwer_best" ] || exit 1
  fi
done

find "$exp/" -type f -name 'wer_best' -exec cat {} \;