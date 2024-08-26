#! /usr/bin/env python

# Copyright 2024 Sean Robertson, Michael Ong
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

import numpy as np
import jiwer
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use
import matplotlib.pyplot as plt

from pydrobert.torch.data import read_trn_iter

def boothroyd_func(x, k):
    return x**k

def get_err(ref_file, hyp_file):
    ref_dict = dict(read_trn_iter(ref_file))
    empty_refs = set(key for key in ref_dict if not ref_dict[key])
    if empty_refs:
        print(
            "One or more reference transcriptions are empty: "
            f"{', '.join(empty_refs)}",
            file=sys.stderr,
            end="",
        )
        return 1
    keys = sorted(ref_dict)
    refs = [" ".join(ref_dict[x]) for x in keys]
    del ref_dict

    hyp_dict = dict((k, v) for (k, v) in read_trn_iter(hyp_file))
    if sorted(hyp_dict) != keys:
        keys_, keys = set(hyp_dict), set(keys)
        print(
            f"ref and hyp file have different utterances!",
            file=sys.stderr,
        )
        diff = sorted(keys - keys_)
        if diff:
            print(f"Missing from hyp: " + " ".join(diff), file=sys.stderr)
        diff = sorted(keys_ - keys)
        if diff:
            print(f"Missing from ref: " + " ".join(diff), file=sys.stderr)
        return 1
    hyps = [" ".join(hyp_dict[x]) for x in keys]
    er = jiwer.wer(refs, hyps)
    return(er)

data_dir = sys.argv[1]
zp_errs = []
lp_errs = []
hp_errs = []

out_path = os.path.join(data_dir, "results")
lp_image = os.path.join(data_dir, "lp_zp_graph.png")
hp_image = os.path.join(data_dir, "hp_zp_graph.png")
both_image = os.path.join(data_dir, "overlay_graph.png")
out = open(out_path, "w")

for snr_dir in os.scandir(data_dir):
        if snr_dir.is_dir():
            for _, split_dirs, _ in os.walk(snr_dir):
                for split_dir in split_dirs:
                    ref_file = os.path.join(snr_dir, split_dir, "ref.trn")
                    hyp_file = os.path.join(snr_dir, split_dir, "hyp.trn")
                    if split_dir == "3":
                        zp_errs.append(get_err(ref_file,hyp_file))
                    elif split_dir == "2":
                        lp_errs.append(get_err(ref_file,hyp_file))
                    elif split_dir == "1":
                        hp_errs.append(get_err(ref_file,hyp_file))

lz_k = curve_fit(boothroyd_func, xdata= zp_errs, ydata= lp_errs)[0][0]
hz_k = curve_fit(boothroyd_func, xdata= zp_errs, ydata= hp_errs)[0][0]

out.write(f"LP/ZP k value:\t{lz_k}\n")
out.write(f"HP/ZP k value:\t{hz_k}\n")
out.close

plt.figure(figsize = (10,8))
plt.plot(zp_errs, lp_errs, 'bo')
xseq = np.linspace(0, 1, num=100)
plt.plot(xseq, xseq**lz_k, 'r')
plt.xlabel('ZP error rate')
plt.ylabel('LP error rate')
plt.savefig(lp_image)

plt.figure(figsize = (10,8))
plt.plot(zp_errs, hp_errs, 'b+')
xseq = np.linspace(0, 1, num=100)
plt.plot(xseq, xseq**hz_k, 'g')
plt.xlabel('ZP error rate')
plt.ylabel('HP error rate')
plt.savefig(hp_image)

plt.figure(figsize = (20,16))
plt.plot(zp_errs, lp_errs, 'bo')
plt.plot(zp_errs, hp_errs, 'b+')
xseq = np.linspace(0, 1, num=100)
plt.plot(xseq, xseq**lz_k, 'r')
plt.plot(xseq, xseq**hz_k, 'g')
plt.xlabel('ZP error rate')
plt.ylabel(' error rate')
plt.savefig(both_image)