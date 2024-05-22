#! /usr/bin/env bash

for part in test train dev; do
    for layer in 12 24 36 46 47 48; do
        mkdir -p data/mms_hidden/layer_$layer/$part/feat/
        for file in data/mms_hidden/$part/*$layer.pt; do
            filename="$(basename "$file")"
            out_file=data/mms_hidden/layer_$layer/$part/feat/$filename
            python3 squeeze_mms.py "$file" "$out_file"
            rm -f "$file"
        done
    done
done