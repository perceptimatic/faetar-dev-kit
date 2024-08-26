#! /usr/bin/env bash

if [ $# -ne 4 ]; then
    echo "Usage: $0 perp_file hypothesis_trn_file num-to-split out-dir"
    exit 1
fi

perp_file="$1"
hyp_trn="$2"
ns="$3"
out_dir="$4"

if [ ! -f "$perp_file" ]; then
    echo "'$perp_file' is not a file"
    exit 1
fi

if [ ! -f "$hyp_trn" ]; then
    echo "'$hyp_trn' is not a file"
    exit 1
fi

if [ ! -d "$out_dir" ]; then
    echo "'$out_dir' is not a directory"
    exit 1
fi

set -eo pipefail

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

for i in $(seq 1 $ns); do
    mkdir -p "$out_dir/$i"
done

awk '{print $0"\t"NR}' "$perp_file" |
sort -n -k 1,1 |
awk -v ns="$ns" -v lines="$(wc -l < "$perp_file")" -v filename="$(basename "$perp_file")" -v out_dir="$out_dir" \
'BEGIN {
    FS = "\t";
    OFS = " ";
}

NR == FNR {
    cut = 0.05;

    for (i = 1; i <= ns; i++) {
        out_ref_path = out_dir "/" i "/ref.trn"
        if ((NR >= (lines * cut)) && (NR <= (lines * (1 - cut)))) {
            if ((NR > ((lines * (1 - 2 * cut)) * ((i - 1) / ns) + (lines * cut))) && (NR <= ((lines * (1 - 2 * cut)) * (i / ns) + (lines * cut)))) {
                print $4 "\t" $2, $3 > out_ref_path;
                ind_name = sprintf("%s_%s", $4, i);
                extract[ind_name] = "x";
            }
        }
    }
}

NR != FNR {
    for (i = 1; i <= ns; i++) {
        out_hyp_path = out_dir "/" i "/hyp.trn"
        for (ind_name in extract) {
            if (ind_name == FNR "_" i) {
                print $0 > out_hyp_path;
                delete extract[ind_name];
                next;
            }
            else {
                continue
            }
        }
    }
}' "-" "$hyp_trn"

for file in "$out_dir"/*/hyp.trn; do
    split_text "$file" 
done

for file in "$out_dir"/*/ref.trn; do
    sort -nk 1,1 "$file" |
    awk 'BEGIN {FS = "\t"} {print $2}' > "$file"_
    mv "$file"{_,}
done