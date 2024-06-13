#! /usr/bin/env python

# Copyright 2024 Sean Robertson
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

import sys
import argparse

import numpy as np
import jiwer

from pydrobert.torch.data import read_trn_iter

def get_ln_err(ref_file, hyp_file):
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

    hname = hyp_file.name
    hyp_dict = dict((k, v) for (k, v) in read_trn_iter(hyp_file))
    if sorted(hyp_dict) != keys:
        keys_, keys = set(hyp_dict), set(keys)
        print(
            f"ref and hyp file '{hname}' have different utterances!",
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
    return(np.log(er))


def main(args=None):

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "ref_file", type=argparse.FileType("r"), help="The reference trn file"
    )
    parser.add_argument(
        "hyp_file", type=argparse.FileType("r"), help="The hypothesis trn file",
    )
    parser.add_argument(
        "ref_zp_file", type=argparse.FileType("r"), help="The reference zp trn file"
    )
    parser.add_argument(
        "hyp_zp_file", type=argparse.FileType("r"), help="The hypothesis trn file",
    )
    options = parser.parse_args(args)

    ap_ln_err = get_ln_err(options.ref_file, options.hyp_file)
    zp_ln_err = get_ln_err(options.ref_zp_file, options.hyp_zp_file)

    print("k: " + str(ap_ln_err / zp_ln_err))

    exit()

if __name__ == "__main__":
    sys.exit(main())
