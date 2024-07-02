# specify what folders to work with

# copy lstm trn file to folder

#! /usr/bin/env bash

# Copyright 2024 Sean Robertson, Michael Ong
# Apache 2.0

export PYTHONUTF8=1
[ -f "path.sh" ] && . "path.sh"

usage="Usage: $0 [-h] [-p] [-S INT] [-m DIR] [-d DIR] [-P FILE] [-o DIR] [-n DIR] [-N DIR] [-L INT] [-H INT] 
[-w NAT] [-a NAT] [-B NAT] [-l NAT]"
pointwise=false
point_snr=
model=exp/mms_lsah_q
data=
perplexity_lm=
partitions=(dev)          # partitions to perform
out_dir=data/boothroyd
norm_wavs_out_dir=norm
noise_wavs_out_dir=noise
snr_low=-10
snr_high=30
width=100
alpha_inv=1
beta=1
lm_ord=0
help="Determine the k value for a given model

Options
    -h          Display this help message and exit
    -p          Determine the k-value at a for a specific SNR
    -s INT      The SNR (in dB) used if -p is set to true (default: '$point_snr')
    -m DIR/{lstm, mms_lsah, mms_lsah_q}
                The model directory (default: '$model')
    -d DIR      The data directory (default: '$data_dir')
    -P FILE     The language model used to calculate the perplexity (default: '$perplexity_lm')
    -o DIR      The output directory (default: '$out_dir')
    -n DIR      The output subdirectory for the normalized wav files (default: '$out_dir/$norm_wavs_out_dir')
    -N DIR      The output subdirectory for the normalized wav files
                with added noise (default: '$out_dir/$norm_wavs_out_dir')
    -L INT      The lower bound (inclusive) of signal-to-noise ratio (SNR) in dB (default: '$snr_low')
    -H INT      The upper bound (inclusive) of signal-to-noise ratio (SNR) in dB (default: '$snr_high')
    -w NAT      pyctcdecode's beam width (default: $width)
    -a NAT      pyctcdecode's alpha, inverted (default: $alpha_inv)
    -B NAT      pyctcdecode's beta (default: $beta)
    -l NAT      n-gram LM order. 0 is greedy; 1 is prefix with no LM (default: $lm_ord)"

while getopts "hps:m:d:P:o:n:N:L:H:w:a:B:l:" name; do
    case $name in
        h)
            echo "$usage"
            echo ""
            echo "$help"
            exit 0;;
        p)
            pointwise=true;;
        s)
            point_snr="$OPTARG";;
        m)
            model="$OPTARG";;
        d)
            data="$OPTARG";;
        P)
            perplexity_lm="$OPTARG";;
        o)
            out_dir="$OPTARG";;
        n)
            norm_wavs_out_dir="$OPTARG";;
        N)
            noise_wavs_out_dir="$OPTARG";;
        L)
            snr_low="$OPTARG";;
        H)
            snr_high="$OPTARG";;
        w)
            width="$OPTARG";;
        a)
            alpha_inv="$OPTARG";;
        B)
            beta="$OPTARG";;
        l)
            lm_ord="$OPTARG";;
        *)
            echo -e "$usage"
            exit 1;;
    esac
done
shift $(($OPTIND - 1))
if $pointwise; then
  if ! [[ "$point_snr" =~ ^-?[0-9]+\.?[0-9]*$ ]] 2> /dev/null; then
      echo -e "$point_snr is not a real number! set -s appropriately, or add a leading zero!"
      exit 1
  fi
fi
if [ ! -d "$model" ]; then
    echo -e "'$model' is not a directory! set -m appropriately!"
    exit 1
fi
if [ ! -d "$data" ]; then
    echo -e "'$data' is not a directory! set -d appropriately!"
    exit 1
fi
if [ ! -f "$perplexity_lm" ]; then
    if [ ! $lm_ord -ge 2 ]; then
      echo -e "'$perplexity_lm' is not a file! set -P appropriately!"
      echo -e "If you want to use the language model used for decoding, set -l to be greater than 1"
      exit 1
    fi
    echo -e "'$perplexity_lm' is not a file, but -l is greater than 1,"
    echo -e "so the perplexity calculation will use the language model used in decoding."
fi
if ! mkdir -p "$out_dir" 2> /dev/null; then
    echo -e "Could not create '$out_dir'! set -o appropriately!"
    exit 1
fi
if ! mkdir -p "$out_dir/$norm_wavs_out_dir" 2> /dev/null; then
    echo -e "Could not create '$out_dir/$norm_wavs_out_dir'! set -n appropriately!"
    exit 1
fi
if ! mkdir -p "$out_dir/$noise_wavs_out_dir" 2> /dev/null; then
    echo -e "Could not create '$out_dir/$noise_wavs_out_dir'! set -N appropriately!"
    exit 1
fi
if ! [[ "$snr_low" =~ ^-?[0-9]+\.?[0-9]*$ ]] 2> /dev/null; then
    echo -e "$snr_low is not a real number! set -L appropriately, or add a leading zero!"
    exit 1
fi
if ! [[ "$snr_high" =~ ^-?[0-9]+\.?[0-9]*$ ]] 2> /dev/null; then
    echo -e "$snr_high is not a real number! set -H appropriately, or add a leading zero!"
    exit 1
fi
if ! [[ "$(bc -l <<< "$snr_low <= $snr_high")" -ne 0 ]]; then
    echo -e "$snr_low is greater than $snr_high! set -L and -H appropriately!"
    exit 1
fi
if ! [ "$width" -gt 0 ] 2> /dev/null; then
    echo -e "$width is not a natural number! set -w appropriately!"
    exit 1
fi
if ! [ "$alpha_inv" -gt 0 ] 2> /dev/null; then
    echo -e "$alpha_inv is not a natural number! set -a appropriately!"
    exit 1
fi
if (( "$(bc -l <<< "$beta < 0")" )); then
    echo -e "$beta is not greater than 0! set -B appropriately!"
    exit 1
fi
if ! [ "$lm_ord" -ge 0 ] 2> /dev/null; then
    echo -e "$lm_ord is not a non-negative int! set -l appropriately!"
    exit 1
fi

function split_text() {
    file="$1"
    awk \
    'BEGIN {
        FS = " ";
        OFS = " ";
    }

    {
        filename = $NF;
        NF --;
        gsub(/ /, "_");
        gsub(/\[fp\]|d[zʒ]ː|tʃː|d[zʒ]|tʃ|\Sː|\S/, "& ");
        gsub(/ +/, " ");
        print $0 filename;
    }' "$file" > "$file"_
    mv "$file"{_,}
}

set -eo pipefail

boothroyd="$(dirname "$0")"

for part in "${partitions[@]}"; do
  # Data prep -------------------------------------------------
  mkdir -p "$out_dir/$norm_wavs_out_dir/$part"
  mkdir -p "$out_dir/$noise_wavs_out_dir/$part"

  if ! [ -f "$out_dir/$norm_wavs_out_dir/$part/.done" ]; then
    # Normalize data volume to same reference average RMS
    bash "$boothroyd"/normalize_data_volume.sh -d "$data/$part" -o "$out_dir/$norm_wavs_out_dir/$part"
    touch "$out_dir/$norm_wavs_out_dir/$part/.done"
  fi

  if ! [ -f "$out_dir/$noise_wavs_out_dir/$part/trn_lstm" ]; then
    cp "$data/$part/trn" "$out_dir/$noise_wavs_out_dir/$part/trn"
    split_text "$out_dir/$noise_wavs_out_dir/$part/trn"
    mv "$out_dir/$noise_wavs_out_dir/$part/trn"{,_lstm}
  fi

  for snr in $(seq $snr_low $snr_high); do
    spart="$out_dir/$noise_wavs_out_dir/$part/snr${snr}"
    if ! [ -f "$spart/.done_noise" ]; then
      bash "$boothroyd"/add_noise.sh -d "$out_dir/$norm_wavs_out_dir/$part" -s "$snr" \
      -o "$out_dir/$noise_wavs_out_dir" -p "$part"
      touch "$spart/.done_noise"
    fi

    # Decoding -------------------------------------------------

    if [[ "$(basename $model)" == "mms_lsah_q" ]]; then
      if [ "$lm_ord" = 0 ]; then
        name="greedy"
        if  ! [ -f "$model/k_decode/${part}/${name}/snr${snr}_${name}.trn" ]; then
          echo "Greedily decoding '$spart'"
          mkdir -p "$model/k_decode/$part/$name"
          python3 ./mms.py decode \
              "$model" "$spart" "$model/k_decode/${part}/${name}/snr${snr}_${name}.csv_"
          mv "$model/k_decode/${part}/${name}/snr${snr}_${name}.csv"{_,}
          python3 ./mms.py metadata-to-trn \
              "$model/k_decode/${part}/${name}/snr${snr}_${name}."{csv,trn_}
          mv "$model/k_decode/${part}/${name}/snr${snr}_${name}.trn"{_,}
        fi
      else
        if [ ! -f "prep/ngram_lm.py" ]; then
          echo "Initializing Git submodule"
          git submodule update --init --remote prep
        fi
        if  ! [ -f "$model/k_decode/logits/${part}_snr${snr}/.done" ]; then
            echo "Dumping logits of '$spart'"
            mkdir -p "$model/k_decode/logits/${part}_snr${snr}"
            python3 ./mms.py decode --logits-dir "$model/k_decode/logits/${part}_snr${snr}" \
                "$model" "$spart" "/dev/null"
            touch "$model/k_decode/logits/${part}_snr${snr}/.done"
        fi

        if [ "$lm_ord" = 1 ]; then
            name="w${width}_nolm"
            alpha_inv=1
            beta=1
            lm_args=( )
        else
          name="w${width}_lm${lm_ord}_ainv${alpha_inv}_b${beta}"
          lm="$model/lm/${lm_ord}gram.arpa"
          lm_args=( --lm "$lm" )
          if ! [ -f "$lm" ]; then
            echo "Constructing '$lm'"
            mkdir -p "$model/lm"
            python3 ./prep/ngram_lm.py -o $lm_ord -t 0 1 < "etc/lm_text.txt" > "${lm}_"
            mv "$lm"{_,}
          fi
        fi

        if ! [ -f "$model/token2id" ]; then
            echo "Constructing '$model/token2id'"
            python3 ./mms.py vocab-to-token2id "$model/"{vocab.json,token2id}
        fi

        if ! [ -f "$model/k_decode/${part}/${name}/snr${snr}_${name}.trn" ]; then
            echo "Decoding $spart"
            mkdir -p "$model/k_decode/${part}/${name}"
            python3 ./prep/logits-to-trn-via-pyctcdecode.py \
                --char "${lm_args[@]}" \
                --words "etc/lm_words.txt" \
                --width $width \
                --beta $beta \
                --alpha-inv $alpha_inv \
                --token2id "$model/token2id" \
                "$model/k_decode/logits/${part}_snr${snr}" "$model/k_decode/${part}/${name}/snr${snr}_${name}.trn"
        fi
      fi
    fi

    # k calculation -------------------------------------------------
    # make sure to set lm above when decoding at some point for -l >= 2
    if [ ! -f "$perplexity_lm" ]; then
      perplexity_lm=$lm
    fi

    perplexity_filename="$out_dir/$noise_wavs_out_dir/$part/perplexity_$(basename $perplexity_lm)"

    if [ ! -f "$perplexity_filename" ]; then
      python3 "$boothroyd"/get_perplexity.py "$perplexity_lm" "$out_dir/$noise_wavs_out_dir/$part/trn_lstm"
    fi

    if [ ! -f "$spart/.done_split" ]; then
      bash "$boothroyd"/section_data.sh \
      "$perplexity_filename" \
      "$model/k_decode/${part}/${name}/snr${snr}_${name}.trn" 3 "$spart"
      echo -e "split using: $perplexity_filename" > "$spart/.done_split"
    fi
  done

  python3 "$boothroyd"/get_snr_k.py "$out_dir/$noise_wavs_out_dir/$part"
done

exit 4



























# data=data/boothroyd
# model="$1"
# exp_name="$2"

# for x in test train dev; do
#     mkdir -p "$data"/"$x"_"$model"

#     cp data/lstm/"$x"/trn_lstm "$data"/"$x"_"$model"/trn_lstm

#     python3 faetar_dev_kit/boothroyd/get_perplexity.py \
#     exp/mms_lsah_q/lm/5gram.arpa "$data"/"$x"_"$model"/trn_lstm 

#     faetar_dev_kit/boothroyd/section_data.sh \
#     "$data"/"$x"_"$model"/perplexity_5gram.arpa exp/"$exp_name"/decode/"$x"_w100_unlablm5_ainv1_b1.trn 3
    
#     echo "hp/zp $x"
#     python3 faetar_dev_kit/boothroyd/get_single_k.py \
#     "$data"/"$x"_"$model"/1_ref_perplexity_5gram.arpa "$data"/"$x"_"$model"/1_hyp_perplexity_5gram.arpa \
#     "$data"/"$x"_"$model"/3_ref_perplexity_5gram.arpa "$data"/"$x"_"$model"/3_hyp_perplexity_5gram.arpa

#     echo "lp/zp $x"
#     python3 faetar_dev_kit/boothroyd/get_single_k.py \
#     "$data"/"$x"_"$model"/2_ref_perplexity_5gram.arpa "$data"/"$x"_"$model"/2_hyp_perplexity_5gram.arpa \
#     "$data"/"$x"_"$model"/3_ref_perplexity_5gram.arpa "$data"/"$x"_"$model"/3_hyp_perplexity_5gram.arpa

# done

