#! /usr/bin/env bash

for part in test train dev; do
    for layer in 12 24 36 46 47 48; do
        mkdir -p data/mms_hidden/layer_$layer/$part/feat/
        python3 squeeze_mms.py "data/mms_hidden/$part/" "$layer" "$part"
    done
done