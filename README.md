# faetar-dev-kit
Data processing and baselines for the 2025 Faetar Low-Resource ASR Challenge

See the [Challenge website](https://perceptimatic.github.io/faetarspeech/) for more information, as well as the [paper documenting the benchmark corpus](https://arxiv.org/abs/2409.08103)

## Installation & activation

### Conda

``` sh
# installs ALL dependencies in a conda environment and activates it
# (if you're just training or decoding with one baseline, you probably
# don't need all of them)
conda env create -f environment.yaml
conda activate faetar-dev-kit
```

### Pip

``` sh
pip install -r requirements.txt

# If pip cannot find binary wheels, and has to be built from source, an up to
# date Rust Compiler and Arrow installation may be needed as dependencies.
# These will probably be available through your package manager.
```

## ASR baselines

``` sh
# assumes data/ is populated with {mms_lsah, ml_superb,...} partitions which have 
# {train,dev,test,...} subpartitions
# and exp/ contains all artifacts (checkpoints, hypothesis transcriptions, etc.)

# Train and greedily decode MMS-LSAH
# successfully trained on a single T4 core
./run_mms_lsah.sh  # -h flag for options

# Train and greedily decode MMS-10min or MMS-1h
# makes a new virtualenv; won't work on Git Bash
#  (faetar-dev-kit should contain necessary build tools)
# successfully trained on a single A40 core
./run_ml_superb.sh  # 10min
./run_ml_superb.sh -e exp/mms-1h -p 1h # 1h

# compute the PER, differences, and CIs of a model on a partition of the data directory
./evaluate_asr.sh -d data/mms_lsah -p train -e exp/mms_lsah -n 1000  # -h flag for options
```

## Evaluating your own model

Place the decodings for each model in a sudirectory of a directory called decodings. (if decodings has no subdirectories it is assumed that the decodings were created by only one model)

The decodings should be named {partition}_*.trn.
The format of the trn files should have on each line: <transcription> (<file_id>).

To obtain the evaluation metrics, use the helper script `summary.sh`
``` sh
./summary.sh decodings/ data/
```

## License and attribution

The *MMS-LSAH* baseline adapts the excellent MMS fine-tuning [blog
post](https://huggingface.co/blog/mms_adapters) by Patrick von Platen to the
challenge. I could not see any license information in the MMS blog post.

[ESPNet](https://github.com/espnet/espnet/tree/master) is [Apache
2.0](./LICENSE) licensed. The forked version of ESPNet used in the *MMS-10min*
and *MMS-1h* baselines removes most of the recipes besides
`espnet/egs2/ml_superb`. In `espnet/egs2/TEMPLATE/asr1/db.sh`, `MLSUPERB` was
set to `downloads`. `espnet/egs2/ml_superb/asr1/local/single_lang_data_prep.py`
was modified to handle Faetar. Config files were copied from ESPNet into this
repository, modified to point to MMS. Finally, `run_ml_superb.sh` is very
loosely based on `espnet/egs2/ml_superb/asr1/run_mono.sh`.

This code is licensed with [Apache 2.0](./LICENSE).
