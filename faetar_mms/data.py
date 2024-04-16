# Copyright 2024 Sean Robertson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# load_partition was adapted from Patrick von Platen's
#
#   https://huggingface.co/blog/mms_adapters
#
# last accessed April 15th, 2024

from typing import Literal
from pathlib import Path

from datasets import load_dataset, Audio, Dataset
from transformers import Wav2Vec2Processor

from .args import Options

def load_partition(
    options: Options,
    part: Literal["train", "dev", "decode"],
    processor: Wav2Vec2Processor,
) -> Dataset:
    if part == "train":
        data = options.train_data
    elif part == "dev":
        data = options.dev_data
    else:
        data = options.decode_data
    data = data.absolute()

    ds = load_dataset("audiofolder", data_dir=data, split="all")
    ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

    def filter_short(batch):
        audio = batch["audio"]
        duration = len(audio["array"]) / audio["sampling_rate"]
        return duration > 0.5

    def prepare_dataset(batch):
        audio = batch["audio"]
        if part == "decode":
            batch["file_name"] = Path(audio["path"]).relative_to(data).as_posix()

        # batched output is "un-batched"
        batch["input_values"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_values[0]
        batch["input_length"] = len(batch["input_values"])

        batch["labels"] = processor(text=batch["sentence"]).input_ids
        return batch
    
    if part != "decode":
        ds = ds.filter(filter_short)
    ds = ds.map(prepare_dataset, remove_columns=ds.column_names)
    return ds
