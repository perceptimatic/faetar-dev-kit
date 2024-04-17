# mms-faetar
Baseline implementation for Faetar grand challenge

Adapts the excellent MMS fine-tuning [blog
post](https://huggingface.co/blog/mms_adapters) by Patrick von Platen to the
challenge. Uses Python scripts, not notebooks, because we're not savages.

## Installation & activation

``` sh
conda env create -f environment.yaml
git submodule init  # initialize 'pytorch-database-prep' into 'prep'
conda activate faetar-mms
export PYTHONUTF8=1
```

## Training and evaluation (basic)

``` sh
data=data
exp=exp

mkdir -p "$exp/decode"

# construct metadata.csv for each partition
for d in "$data/"{dev,test,train}; do
    find "$d" -name '*.txt' -printf '%P,' -exec cat {} \; |
        sed 's/\.txt,/.wav,/' |
        sort |
        cat <(echo "file_name,sentence") - > "$d/metadata.csv"
done

# construct vocab.json
./run.py write-vocab "$data/train/metadata.csv" "$exp/vocab.json"

# train the model
./run.py train "$exp/vocab.json" "$data/"{train,dev} "$exp"

# greedy decoding
./run.py decode "$exp" "$data/test" "$exp/decode/test_greedy.csv"

# compute per
./run.py evaluate "$data/test/metadata.csv" "$exp/decode/test_greedy.csv"
```

## Advanced evaluation

``` sh
# continues on from training
lm_orders=( 0 2 3 4 5 6 )
width=100
alpha_invs=( 1 2 3 4 )
betas=( 1 2 3 4 )

# create token2id fiel from vocab.json
./run.py vocab-to-token2id "$exp/"{vocab.json,token2id}

# build n-gram lm
## FIXME(sdrobert): use all training text that we have
mkdir -p "$exp/lm"
cut -d ',' -f 2 "$data/train/metadata.csv" |
    tee "$exp/lm/text" |
    tr ' ' $'\n' |
    sort -u |
    sed '/^$/d' > "$exp/lm/words.txt"
# you can use kenlm if you prefer
for lm_order in "${lm_orders[@]}"; do
    if [ $lm_order -ne 0 ]; then
        ./prep/ngram_lm.py -o $lm_order -t 0 1 < "$exp/lm/text" > "$exp/lm/lm.$lm_order.arpa"
    fi
done

filter() {
    fn="$1"
    sed 's/\[[^]][^]]*\] //g; s/<[^>][^>]*> //g;' "$fn" > "${fn}_filt"
    ./prep/word2subword.py "$fn"{_filt,_char}
    sed 's/_ //g' "${fn}_char" > "${fn}_phone"
}

# write references as trn files
for x in train dev test; do
    ./run.py metadata-to-trn "$data/$x/"{metadata.csv,ref.trn}
    filter "$data/$x/ref.trn"
done

# dump logits to folder and write hypotheses as trn files
for x in dev test; do
    mkdir -p "$exp/decode/logits/$x"
    # ./run.py decode --logits-dir "$exp/decode/logits/$x" "$exp" "$data/$x" "$exp/decode/${x}_greedy.csv"
    ./run.py metadata-to-trn "$exp/decode/${x}_greedy."{csv,trn}
    filter "$exp/decode/${x}_greedy.trn"
done

# decode dev with a big ol' grid
for lm_order in "${lm_orders[@]}"; do
    for alpha_inv in "${alpha_invs[@]}"; do
        for beta in "${betas[@]}"; do
            name="width${width}_lm${lm_order}_beta${beta}_alpha_inv${alpha_inv}"
            args="$exp/decode/$name.args"
            echo "--char
--words
$exp/lm/words.txt
--width
$width
--beta
$beta
--alpha-inv
$alpha_inv
--token2id
$exp/token2id" > "$args"
            if [ "$lm_order" -ne 0 ]; then
                echo "--lm
$exp/lm/lm.$lm_order.arpa" >> "$args"
            fi
            ./prep/logits-to-trn-via-pyctcdecode.py --batch-size 32 "@$args" "$exp/decode/"{logits/dev,dev_$name.trn}
            filter "$exp/decode/dev_$name.trn"
        done
    done
done

# compute all error rates on dev partition
./prep/error-rates-from-trn.py "$data/dev/ref.trn_filt" "$exp/decode/dev_"*.trn_filt > "$exp/decode/dev_wer"
./prep/error-rates-from-trn.py "$data/dev/ref.trn_char" "$exp/decode/dev_"*.trn_char > "$exp/decode/dev_cer"
./prep/error-rates-from-trn.py "$data/dev/ref.trn_phone" "$exp/decode/dev_"*.trn_phone > "$exp/decode/dev_per"

# use best PER on dev parition's args as test args
# (this won't work if greedy is best)
cp "$exp/decode/$(tail -n 1 $exp/decode/dev_per | cut -d "'" -f 2 | sed 's:.*/dev_\([^.]*\).*:\1:').args" "$exp/decode/best.args"

# compute test hypotheses
./prep/logits-to-trn-via-pyctcdecode.py --batch-size 32 "@$exp/decode/best.args" "$exp/decode/"{logits/test,test_best.trn}
filter "$exp/decode/test_best.trn"

# compute all error rates on test partition
./prep/error-rates-from-trn.py "$data/test/ref.trn_filt" "$exp/decode/test_"*.trn_filt > "$exp/decode/test_wer"
./prep/error-rates-from-trn.py "$data/test/ref.trn_char" "$exp/decode/test_"*.trn_char > "$exp/decode/test_cer"
./prep/error-rates-from-trn.py "$data/test/ref.trn_phone" "$exp/decode/test_"*.trn_phone > "$exp/decode/test_per"
```

## License

This code is licensed with [Apache 2.0](./LICENSE). I could not see any license
information in the blog post.

