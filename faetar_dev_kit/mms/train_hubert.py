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

# much of this code was adapted from Patrick von Platen's
#
#   https://huggingface.co/blog/mms_adapters
#
# last accessed April 15th, 2024

import sys
import json

from typing import Any, Union
from dataclasses import dataclass

import torch
import numpy as np

from safetensors.torch import save_file as safe_save_file
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    HubertForCTC,
    TrainingArguments,
    Trainer,
)
from evaluate import load

from .args import Options
from .data import load_partition

wer_metric = load("wer")


@dataclass
class TrainingRoutines:

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def collate(
        self, features: list[dict[str, Union[list[int], torch.Tensor]]]
    ) -> dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch

    def compute_metrics(self, pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id

        pred_str = self.processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = self.processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}


def train(options: Options):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        options.pretrained_model_id,
    )
    
    tokenizer = Wav2Vec2CTCTokenizer(
        options.vocab_json,
        unk_token=options.unk,
        pad_token=options.pad,
        word_delimiter_token=options.word_delimiter,
        target_lang="fae",
    )

    processor = Wav2Vec2Processor(feature_extractor, tokenizer)

    with open(options.wav2vec2_kwargs_json) as fp:
        wav2vec2_kwargs: dict[str, Any] = json.load(fp)

    for name, expected in (
        ("pad_token_id", processor.tokenizer.pad_token_id),
        ("vocab_size", len(processor.tokenizer)),
    ):
        if name not in wav2vec2_kwargs:
            wav2vec2_kwargs[name] = expected
        elif wav2vec2_kwargs[name] != expected:
            print(
                f"'{options.wav2vec2_kwargs_json}' contains the entry "
                f"'{name}' = {wav2vec2_kwargs[name]}, but we expected "
                f"{expected} based on '{options.vocab_json}'. "
                "You probably shouldn't specify this entry in the first place",
                file=sys.stderr,
            )
            return 1
    if "ignore_mismatched_sizes" not in wav2vec2_kwargs:
        wav2vec2_kwargs["ignore_mismatched_sizes"] = True

    model = HubertForCTC.from_pretrained(
        options.pretrained_model_id, **wav2vec2_kwargs
    )

    dev = load_partition(options, "dev", processor)
    train = load_partition(options, "train", processor)

    with open(options.training_kwargs_json) as fp:
        training_kwargs = json.load(fp)

    training_args = TrainingArguments(output_dir=options.model_dir, **training_kwargs)

    training_routines = TrainingRoutines(processor, padding=True)

    trainer = Trainer(
        model=model,
        data_collator=training_routines.collate,
        args=training_args,
        compute_metrics=training_routines.compute_metrics,
        train_dataset=train,
        eval_dataset=dev,
        # the encoder 'tokenizer', which is the feature extractor
        tokenizer=processor.feature_extractor,
    )

    try:
        trainer.train(resume_from_checkpoint=True)
    except ValueError:  # no checkpoint
        trainer.train()

    trainer.save_model(options.model_dir)
    processor.tokenizer.save_pretrained(options.model_dir)

    return 0
