#! /usr/bin/env python

import sys
import logging

import numpy as np

import argparse

from pydrobert.kaldi.io import open as kaldi_open
from pydrobert.kaldi.io.argparse import KaldiParser
from pydrobert.kaldi.logging import kaldi_logger_decorator
from pydrobert.kaldi.logging import register_logger_for_kaldi
from pydrobert.kaldi.io.util import parse_kaldi_output_path
from pydrobert.kaldi.io.enums import TableType


@kaldi_logger_decorator
def main(args=None):
    """\
Write power of signals to file

A signal's power x is computed as

    P = sum_t (x(t))^2

This program reads 
"""

    logger = logging.getLogger(sys.argv[0])
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    register_logger_for_kaldi(logger)
    parser = KaldiParser(description=main.__doc__, logger=logger)
    parser.add_argument("--channel", type=int, default=-1)
    parser.add_argument(
        "--normalize",
        type="kaldi_bool",
        default=False,
        help="Whether to normalize by the number of samples",
    )
    parser.add_argument(
        "--text",
        type="kaldi_bool",
        default=False,
        help="Write power as text (wxfilename only)",
    )
    parser.add_argument(
        "--magnitude",
        type="kaldi_bool",
        default=False,
        help="Write sqrt of power instead",
    )
    parser.add_argument(
        "--regions",
        type="kaldi_rspecifier",
        default=None,
        help="rspecifier of regions of utts to calculate power of (in secs)",
    )
    scale_by_grp = parser.add_mutually_exclusive_group()
    scale_by_grp.add_argument("--scale-by", type=float, default=1)
    scale_by_grp.add_argument("--inv-scale-by", type=float, default=None)
    parser.add_argument("wav", type="kaldi_rspecifier", help="rspecifier of wavs")
    parser.add_argument("pow", help="wspecifier/wxfilename of power")

    options = parser.parse_args(args)

    if options.inv_scale_by is not None:
        options.scale_by = 1 / options.inv_scale_by

    is_table = parse_kaldi_output_path(options.pow)[0] != TableType.NotATable
    if is_table:
        logger.info("Writing per-utterance power")
    else:
        logger.info("Writing total power")

    wav_reader = kaldi_open(options.wav, "wm", "r", value_style="bs")
    if options.regions is not None:
        region_reader = kaldi_open(options.regions, "ipv", "r+")
    else:
        region_reader = None
    writer = kaldi_open(options.pow, "b", "w", header=not options.text)

    utt_no = -1
    total_power = total_samples = 0
    for utt_no, (utt, (wav, samp_rate)) in enumerate(wav_reader.items()):
        if options.channel == -1 and wav.shape[0] != 1:
            logger.error(f"{utt}: expected mono; got {wav.shape[0]} channels")
        wav = np.square(wav[options.channel], dtype=np.float64)
        n_wav = len(wav)
        if region_reader is None:
            power = float(wav.sum())
            samples = n_wav
        else:
            power = samples = 0
            for start, end in region_reader[utt]:
                if start < 0:
                    logger.error(f"{utt}: region [{start}, {end}) starts before 0!")
                start, end = int(start * samp_rate), int(end * samp_rate)
                if end > n_wav:
                    logger.warning(
                        f"utt {utt}: region [{start / samp_rate}, {end / samp_rate}) "
                        f"ends after wav {n_wav / samp_rate}"
                    )
                    end = n_wav
                if start >= end:
                    logger.warning(
                        f"utt {utt}: region [{start / samp_rate}, {end / samp_rate}) "
                        "is empty"
                    )
                else:
                    power += float(wav[start:end].sum())
                    samples += end - start
        power *= options.scale_by**2
        if not samples:
            logger.warning(
                f"utt {utt} has no samples to compute power over; setting to 0"
            )
        else:
            logger.info(
                f"utt {utt} has power {power} over {samples} samples "
                f"(avg. {power / samples})"
            )
            total_power += power
            total_samples += samples
            if options.normalize:
                power /= samples
            if options.magnitude:
                power **= 0.5
        if is_table:
            writer.write(utt, power)

    logger.info(f"Processed {utt_no + 1} entries")

    if not is_table:
        if options.normalize:
            total_power /= total_samples
        if options.magnitude:
            total_power **= 0.5
        writer.write(total_power, "b", write_binary=not options.text)


if __name__ == "__main__":
    main()