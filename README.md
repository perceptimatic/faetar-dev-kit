# mms-faetar
Baseline implementation for Faetar grand challenge


## Installation

``` sh
conda env create -f environment.yaml
```

## Running

``` sh
data=data

conda activate faetar-mms

# construct metadata.csv for each partition
for d in "$data/"{train,dev,test}; do
    find "$d" -name '*.txt' -printf '%p,' -exec cat {} \; |
        sed 's/\.txt,//' |
        sort |
        cat <(echo "file,sentence") - > "$d/metadata.csv"
done

# construct vocabulary.json
./run.py "$data/train/metadata.csv" vocab.json
```