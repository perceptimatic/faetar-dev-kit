# mms-faetar
Baseline implementation for Faetar grand challenge


## Installation

``` sh
conda env create -f environment.yaml
```

## Running

``` sh
data=data
exp=exp

conda activate faetar-mms

mkdir -p "$exp"

# construct metadata.csv for each partition
for d in "$data/"{dev,test,train}; do
    find "$d" -name '*.txt' -printf '%P,' -exec cat {} \; |
        sed 's/\.txt,/,/' |
        sort |
        cat <(echo "file,sentence") - > "$d/metadata.csv"
done

# construct vocab.json
./run.py write-vocab "$data/train/metadata.csv" "$exp/vocab.json"
```