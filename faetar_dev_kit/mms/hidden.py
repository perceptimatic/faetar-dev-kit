# Code from Robin Huo

import argparse
from glob import glob
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

SEED = 3939

SR = 16000

LAYERS = (
    "z",
    *[f"c{n}" for n in range(48)]
)


def collate(args: Sequence[dict]) -> dict:
    return {
        "audio": {
            "path": tuple(x["audio"]["path"] for x in args),
            "array": tuple(x["audio"]["array"] for x in args),
            "sampling_rate": tuple(x["audio"]["sampling_rate"] for x in args),
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute representations of audio from a pretrained wav2vec2 model."
    )
    parser.add_argument("model_dir", type=Path, help="Pretrained model directory")
    parser.add_argument("input_dir", type=Path, help="Root directory of input files")
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=Path("out"),
        help="Root output directory",
    )
    parser.add_argument(
        "-l",
        "--layer",
        choices=LAYERS,
        nargs="+",
        default=None,
        help="Layer(s) to extract from",
    )
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="DO NOT USE")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_dataset(
        "audiofolder",
        data_files={
            "validation": sorted([str(p.resolve()) for d in ["dev", "test", "train"]
                                  for p in args.input_dir.rglob(f"{d}/*.wav")])
        },
        drop_labels=True,
    ).with_format("numpy")

    model = Wav2Vec2ForCTC.from_pretrained(args.model_dir).to(device)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_dir)

    if args.layer is None:
        layers = frozenset(
            {"z"} | {f"c{i}" for i in range(model.config.num_hidden_layers)}
        )
    elif not args.layer:
        raise ValueError("No layers selected")
    else:
        layers = frozenset(args.layer)

    model.eval()
    with torch.no_grad():
        for b, batch in tqdm(
            enumerate(
                DataLoader(
                    data["validation"],
                    batch_size=args.batch_size,
                    collate_fn=collate,
                )
            )
        ):
            assert all(
                sr == SR for sr in batch["audio"]["sampling_rate"]
            ), f"Unexpected sampling rate found in {batch['audio']['sampling_rate']}"
            input_values = feature_extractor(
                batch["audio"]["array"],
                return_tensors="pt",
                padding=True,
                sampling_rate=SR,
            ).input_values.to(device)
            out_dict = model(input_values, output_hidden_states=True)

            for i in range(args.batch_size):
                for layer in layers:
                    if layer == "z":
                        out = out_dict["hidden_states"][0]
                    elif layer[0] == "c":
                        n = int(layer[1:])
                        out = out_dict["hidden_states"][n + 1]
                    else:
                        raise ValueError(f"Invalid layer: {layer}")
                    out = out[i].numpy(force=True)

                    path = Path(batch["audio"]["path"][i]).resolve()
                    relpath = path.relative_to(args.input_dir.resolve())
                    outpath = args.output_dir / layer / relpath.with_suffix(".npy")
                    outpath.parent.mkdir(parents=True, exist_ok=True)
                    np.save(outpath, out)

                    if args.debug:
                        status = {
                            "path": str(outpath),
                            "audio_samples": batch["audio"]["array"][i].shape[0],
                            "input_shape": input_values.shape,
                            "num_padding": torch.sum(input_values[i] == 0.0).item(),
                            "output_shape": out.shape,
                        }
                        tqdm.write(f"{i}: {status}")

            del input_values
            del out_dict
            torch.cuda.empty_cache()

