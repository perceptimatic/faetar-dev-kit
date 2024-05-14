# faetar-dev-kit
Data processing and baselines for the 2024 Faetar Grand Challenge



## Installation & activation

### Conda

``` sh
conda env create -f environment.yaml
conda activate faetar-mms
```

### Pip

First, you should make sure you have a working and up-to-date Rust installation
(`cargo -V`), since it is needed to compile some of the dependencies. 

``` sh
pip install -r requirements.txt
```

## Running (basic)

The following sequence of commands fine-tunes MMS on the data in `data/train`
and `data/dev`, greedily decodes the data in `data/test`, and compares the
resulting hypothesis transcripts to the reference transcripts in `data/test`,
printing a Phone Error Rate (PER).

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

## License and attribution

The MMS baseline adapts the excellent MMS fine-tuning [blog
post](https://huggingface.co/blog/mms_adapters) by Patrick von Platen to the
challenge. We use Python scripts, not notebooks, because we're not savages.

This code is licensed with [Apache 2.0](./LICENSE). I could not see any license
information in the blog post.

