#! /usr/bin/env bash

# Copyright 2024 Michael Ong
# Apache 2.0

corpus_dir="$1"
train_dir="train"
small_train_dir="10min"
medium_train_dir="1h"
reduced_train_dir="reduced_train"
dirty_train_dir="dirty_data_train"

test_files="he014 he016 he018 he019 hl011 hl027 hl042 hl046 hl053 hl112 hl124 hl153 hl158 hl172"
dev_files="he011 hl045 hl153 hl154"
small_train_spks="heM002 heF001 hlF026"
small_train_files="he007 he011 hl050"
medium_train_spks="heM001 heM002 heM003 heM004 heM008 heM009 heF001 hlF023 hlF026 hlF032 hlM011"
medium_train_files="he007 he010 he011 he014 he016 he022 he030 hl044 hl050 hl051 hl067 hl086 hl087"

cd "$corpus_dir"
mkdir -p "$small_train_dir"
mkdir -p "$medium_train_dir"
mkdir -p "$reduced_train_dir"
mkdir -p "$dirty_train_dir"

find "$train_dir" -type f -print |
grep -f <(tr ' ' '\n' <<< "$small_train_spks") "-" |
grep -f <(tr ' ' '\n' <<< "$small_train_files") "-" |
xargs -I{} bash -c 'cp -R "$1" '"$small_train_dir"'' -- "{}"

find "$train_dir" -type f -print |
grep -f <(tr ' ' '\n' <<< "$medium_train_spks") "-" |
grep -f <(tr ' ' '\n' <<< "$medium_train_files") "-" |
xargs -I{} bash -c 'cp -R "$1" '"$medium_train_dir"'' -- "{}"

find "$train_dir" -type f -print |
comm -23 <(find "$train_dir" -type f -print) <(find "$train_dir" -type f -print | \
grep -f <(tr ' ' '\n' <<< "$medium_train_spks") "-" | \
grep -f <(tr ' ' '\n' <<< "$medium_train_files") "-") |
xargs -I{} bash -c 'cp -R "$1" '"$reduced_train_dir"'' -- "{}"

find "$train_dir" -type f -print |
grep -vf <(tr ' ' '\n' <<< "$test_files") "-" |
grep -vf <(tr ' ' '\n' <<< "$dev_files") "-" |
xargs -I{} bash -c 'cp -R "$1" '"$dirty_train_dir"'' -- "{}"