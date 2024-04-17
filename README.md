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

## Running (basic)

The following sequence of commands fine-tunes a 

``` sh
# assuming data is located in the data/ dir
mkdir -p exp/decode

# construct metadata.csv for each partition
for d in data/{dev,test,train}; do
    ./step.py compile-metadata $d
done

# construct vocab.json
./step.py write-vocab data/train/metadata.csv exp/vocab.json

# train the model
./step.py train exp/vocab.json data/{train,dev} exp

# greedy decoding
./step.py decode exp data/test exp/decode/test_greedy.csv

# compute per
./step.py evaluate data/test/metadata.csv exp/decode/test_greedy.csv
```

## Running (advanced)

It is possible to do prefix search decoding with language model fusion. See
[run.sh](./run.sh) for more details.

## License

This code is licensed with [Apache 2.0](./LICENSE). I could not see any license
information in the blog post.

