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

from typing import Literal, Union
from dataclasses import dataclass

import torch
import numpy as np

from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset, Audio, Dataset
from evaluate import load

from .args import Options

wer_metric = load("wer")


def make_processor(options: Options) -> Wav2Vec2Processor:
    tokenizer = Wav2Vec2CTCTokenizer(
        options.vocab_json,
        unk_token=options.unk,
        pad_token=options.pad,
        word_delimiter_token=options.word_delimiter,
        target_lang=options.iso,
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=options.sampling_rate,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    return Wav2Vec2Processor(feature_extractor, tokenizer)


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

    ds = load_dataset("audiofolder", data_dir=data, split="all")
    ds = ds.cast_column("audio", Audio(sampling_rate=options.sampling_rate))

    def prepare_dataset(batch):
        audio = batch["audio"]

        # batched output is "un-batched"
        batch["input_values"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_values[0]
        batch["input_length"] = len(batch["input_values"])

        batch["labels"] = processor(text=batch["sentence"]).input_ids
        return batch

    ds = ds.map(prepare_dataset, remove_columns=ds.column_names)
    return ds


def load_model(options: Options, processor: Wav2Vec2Processor) -> Wav2Vec2ForCTC:
    return Wav2Vec2ForCTC.from_pretrained(
        options.pretrained_model_id,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
    )


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

    processor = make_processor(options)

    dev = load_partition(options, "dev", processor)
    train = load_partition(options, "train", processor)

    model = load_model(options, processor)

    model.init_adapter_layers()
    adapter_weights = model._get_adapters()
    for param in adapter_weights.values():
        param.requires_grad = True

    training_args = TrainingArguments(
        output_dir=options.ckpt_dir,
        group_by_length=True,
        per_device_train_batch_size=32,
        evaluation_strategy="steps",
        num_train_epochs=4,
        gradient_checkpointing=True,
        fp16=True,
        save_steps=200,
        eval_steps=100,
        logging_steps=100,
        learning_rate=1e-3,
        warmup_steps=100,
        save_total_limit=2,
        push_to_hub=False,
    )

    training_routines = TrainingRoutines(processor, padding=True)

    trainer = Trainer(
        model=model,
        data_collator=training_routines.collate,
        args=training_args,
        compute_metrics=training_routines.compute_metrics,
        train_dataset=train,
        eval_dataset=dev,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
