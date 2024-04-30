# faetar-dev-kit
Data processing and baselines for the 2024 Faetar Grand Challenge

## Installation

``` sh
# installs ALL dependencies in a conda environment and activates it
# (if you're just training or decoding with one baseline, you probably
# don't need all of them)
conda env create -f environment.yaml
conda activate faetar-dev-kit
```

## ASR baselines

``` sh
# assumes data/ is populated with {train,dev,test,...} partitions and
# exp/ contains all artifacts (checkpoints, hypothesis transcriptions, etc.)

# Train and greedily decode MMS-LSAH
./run_mms_lsah.sh  # -h flag for options

# compute the PER, differences, and CIs of all models
./evaluate_asr.sh -n 1000  # -h flag for options
```

## License and attribution

The *MMS-LSAH* baseline adapts the excellent MMS fine-tuning [blog
post](https://huggingface.co/blog/mms_adapters) by Patrick von Platen to the
challenge. We use Python scripts, not notebooks, because we're not savages.

This code is licensed with [Apache 2.0](./LICENSE). I could not see any license
information in the blog post.

