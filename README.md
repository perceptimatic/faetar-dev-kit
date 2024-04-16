# mms-faetar
Baseline implementation for Faetar grand challenge

Adapts the excellent MMS fine-tuning [blog
post](https://huggingface.co/blog/mms_adapters) by Patrick von Platen to the
challenge. Uses Python scripts, not notebooks, because we're not savages.

## Installation & activation

``` sh
conda env create -f environment.yaml
conda activate faetar-mms
```

## Running

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
./run.py decode "$exp" "$data/test" "$exp/decode/test.csv"

# compute per
./run.py evaluate "$data/test/metadata.csv" "$exp/decode/test.csv"
```

## License

This code is licensed with [Apache 2.0](./LICENSE). I could not see any
license information in the blog post.
