#! /usr/bin/env python

# code adapted from https://pytorch.org/audio/stable/tutorials/ctc_forced_alignment_api_tutorial.html

import torch
import torchaudio
import torchaudio.functional as F
import warnings

import argparse

from pathlib import Path
from typing import Optional, Sequence, TextIO
from pyctcdecode.alphabet import BLANK_TOKEN_PTN
from tqdm import tqdm

from datasets import load_dataset, Audio, Dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

BLANK_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

def main(args: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Force align a data folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        fromfile_prefix_chars="@",
    )
    parser.add_argument("--lang", default="fae", help="iso 639 code")
    parser.add_argument("--full-timestamp",
                        action="store_true",
                        default=False,
                        help="If this flag is set, then the timestamps given are in relation "
                        "to the full audio file instead of the utterance")
    parser.add_argument(
        "--blank-first",
        action="store_true",
        default=False,
        help="If nothing looking like a blank token can be found in the vocabulary and "
        "this flag is set, ",
    )

    tk2id_group = parser.add_mutually_exclusive_group(required=True)
    tk2id_group.add_argument(
        "--id2token",
        type=argparse.FileType("r"),
        default=None,
        help="Path to id2token.txt file, mapping integer ids to tokens",
    )
    tk2id_group.add_argument(
        "--token2id",
        type=argparse.FileType("r"),
        default=None,
        help="Path to token2id.txt file, mapping tokens to integer ids",
    )

    ddir_arg = parser.add_argument("data_dir", type=Path, help="Data directory")
    mdir_arg = parser.add_argument(
        "model_dir",
        type=Path,
        help="Path to model dir",
    )
    parser.add_argument(
        "tws",
        nargs="?",
        type=argparse.FileType("w"),
        default=argparse.FileType("w")("-"),
        help="The output tws file. Defaults to stdout",
    )

    options = parser.parse_args(args)
    lang: str = options.lang
    full_timestamp: bool = options.full_timestamp
    blank_first: bool = options.blank_first
    data_dir: Path = options.data_dir
    model_dir: Path = options.model_dir
    tws: TextIO = options.tws
    

    def load_partition(
        processor: Wav2Vec2Processor,
    ) -> Dataset:
        data = data_dir.absolute()

        ds = load_dataset("audiofolder", data_dir=data, split="all")
        ds = ds.cast_column(
            "audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate)
        )

        def prepare_dataset(batch):
            audio = batch["audio"]
            batch["file_name"] = Path(audio["path"]).relative_to(data).as_posix()

            # batched output is "un-batched"
            batch["input_values"] = processor(
                audio["array"], sampling_rate=audio["sampling_rate"]
            ).input_values[0]
            batch["input_length"] = len(batch["input_values"])

            if "sentence" in batch:
                batch["labels"] = processor(text=batch["sentence"]).input_ids
            return batch
        
        ds = ds.map(prepare_dataset, remove_columns=ds.column_names)
        return ds

    id2token_file: TextIO
    swap: bool
    if options.id2token is not None:
        id2token_file, swap = options.id2token, False
    else:
        assert options.token2id is not None
        id2token_file, swap = options.token2id, True
    id2tname = id2token_file.name

    if not data_dir.is_dir():
        raise argparse.ArgumentError(ddir_arg, "not a directory")
    
    if not model_dir.is_dir():
        raise argparse.ArgumentError(mdir_arg, "not a directory")

    id2token = dict()
    token2id = dict()
    blank_token = None
    if swap:
        expected = "<token> <integer-id>"
    else:
        expected = "<integer-id> <token>"
    for no, line in enumerate(id2token_file):
        pair = line.strip().split()
        try:
            if swap:
                token, id = pair
            else:
                id, token = pair
            id = int(id)
            assert id >= 0
        except:
            raise ValueError(
                f"could not parse {id2tname} line {no + 1}: expected "
                f"'{expected}', got '{line}'"
            )
        if BLANK_TOKEN_PTN.match(token):
            if blank_token is not None:
                raise ValueError(
                    f"Two tokens {blank_token} and {token} from {id2tname} "
                    "look like blank tokens. pyctcdecode can't handle this"
                )
            blank_token = token
        if id in id2token:
            raise ValueError(f"Duplicate id {id} in {id2tname}")
        if token in token2id:
            raise ValueError(f"Duplicate token {token} in {id2tname}")
        token2id[token], id2token[id] = id, token

    if not id2token:
        raise ValueError(f"{id2tname} is empty!")
    max_vocab = max(id2token)
    assert max_vocab >= 0
    for id in range(max_vocab):
        if id not in id2token:
            raise ValueError(
                f"found id {max_vocab} in {id2tname} but not {id}. Ids need to be "
                "contiguous"
            )
    vocab = [x[1] for x in sorted(id2token.items())]
    if blank_token is None:
        assert BLANK_TOKEN not in vocab
        blank_token = BLANK_TOKEN
        if blank_first:
            warnings.warn(
                f"Adding {blank_token} to the beginning of the vocabulary because "
                "--blank-first was set"
            )
            vocab.insert(0, blank_token)
        else:
            warnings.warn(
                f"Adding {blank_token} to the end of the vocabulary. If it should be "
                "first, set --blank-first"
            )
            vocab.append(blank_token)

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = "cpu"

    model = Wav2Vec2ForCTC.from_pretrained(
        model_dir, target_lang=lang
    ).to(device)
    processor = Wav2Vec2Processor.from_pretrained(
        model_dir, target_lang=lang
    )

    ds = load_partition(processor)

    def align(elem):
        input_dict = processor(
            elem["input_values"],
            sampling_rate=processor.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        logit = model(input_dict.input_values.to(device)).logits.cpu()
        emission = torch.as_tensor(logit).contiguous()
        targets = torch.as_tensor(elem["labels"]).contiguous().unsqueeze(0)
        alignments, scores = F.forced_align(emission, targets, blank=vocab.index(blank_token))
        alignments, scores = alignments[0], scores[0] 
        scores = scores.exp()
        frame_num = emission.size(1)
        wav_length = len(elem["input_values"])
        return alignments, scores, frame_num, wav_length
    
    def unflatten(list_, lengths):
        assert len(list_) == sum(lengths)
        i = 0
        ret = []
        for l in lengths:
            ret.append(list_[i : i + l])
            i += l
        return ret
    
    sample_rate = processor.feature_extractor.sampling_rate

    for elem in tqdm(ds):
        aligned_tokens, alignment_scores, num_frames, wav_length = align(elem)
        # for i, ali in enumerate(aligned_tokens):
        #     print(f"{i:3d}:\t {vocab[ali]}")
        # print("____________________")
        token_spans = F.merge_tokens(aligned_tokens, alignment_scores, blank=vocab.index(blank_token))
        # for token in token_spans:
        #     print(vocab[token.token] + "\t" + str(token.start) + "\t" + str(token.end))
        # print("____________________")
        token_spans = [s for s in token_spans if vocab[s.token] != "_"]
        # for token in token_spans:
        #     print(vocab[token.token] + "\t" + str(token.start) + "\t" + str(token.end))
        # exit()
        transcript = ''.join(["_[fp]_" if vocab[token] == "[fp]" else vocab[token] for token in elem["labels"]]).strip("_").split("_")
        word_spans = unflatten(token_spans, [1 if word == "[fp]" else len(word) for word in transcript])
        for word in range(len(transcript)):
            ratio = wav_length / num_frames
            word_start = int(ratio * word_spans[word][0].start) / sample_rate
            word_end = int(ratio * word_spans[word][-1].end) / sample_rate
            if full_timestamp:
                utt_start = int(Path(elem["file_name"]).stem.split("_")[1]) / 100
                tws.write(f'{Path(elem["file_name"]).stem.replace("_", "-")}\t{word_start + utt_start:.2f}\t{word_end + utt_start:.2f}\t{transcript[word]}\n')
            else:
                tws.write(f'{Path(elem["file_name"]).stem.replace("_", "-")}\t{word_start:.2f}\t{word_end:.2f}\t{transcript[word]}\n')

if __name__ == "__main__":
    main()
