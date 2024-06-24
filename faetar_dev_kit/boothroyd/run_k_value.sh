# specify what folders to work with

# copy lstm trn file to folder

#! /usr/bin/env bash

# Copyright 2024 Sean Robertson, Michael Ong
# Apache 2.0

exp=exp        # experiment directory
data=data/mms_lsah      # data directory
partitions=(unlab_test)          # partitions to perform
out_dir=data/boothroyd          # output directory
norm_wavs_out_dir=norm       # normalized wavs output subdirectory
noise_wavs_out_dir=noise       # normalized wavs + noise output subdirectory
snr_low=-10      # lower bound (inclusive) of signal-to-noise ratio (SNR)
snr_high=30      # upper bound (inclusive) of signal-to-noise ratio (SNR)

. ./path.sh

set -e

mkdir -p "$out_dir/$norm_wavs_out_dir"
mkdir -p "$out_dir/$noise_wavs_out_dir"

boothroyd="$(dirname "$0")"

# Normalize data volume to same reference average RMS
for part in "${partitions[@]}"; do
  bash "$boothroyd"/normalize_data_volume.sh -d "$data/$part" -o "$out_dir/$norm_wavs_out_dir/$part"
done

exit 1


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


























# data=data/boothroyd
# model="$1"

# for x in test train dev; do
#     faetar_dev_kit/boothroyd/section_data.sh \
#     "$data"/"$x"_"$model"/perplexity_5gram.arpa exp/mms_lsah/decode/"$x"_greedy.trn 3
    
#     echo "hp/zp $x"
#     python faetar_dev_kit/boothroyd/get_k.py \
#     "$data"/"$x"_"$model"/1_ref_perplexity_5gram.arpa "$data"/"$x"_"$model"/1_hyp_perplexity_5gram.arpa \
#     "$data"/"$x"_"$model"/3_ref_perplexity_5gram.arpa "$data"/"$x"_"$model"/3_hyp_perplexity_5gram.arpa

#     echo "lp/zp $x"
#     python faetar_dev_kit/boothroyd/get_k.py \
#     "$data"/"$x"_"$model"/2_ref_perplexity_5gram.arpa "$data"/"$x"_"$model"/2_hyp_perplexity_5gram.arpa \
#     "$data"/"$x"_"$model"/3_ref_perplexity_5gram.arpa "$data"/"$x"_"$model"/3_hyp_perplexity_5gram.arpa

# done

