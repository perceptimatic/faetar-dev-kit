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

from typing import Literal, Union
from dataclasses import dataclass

import torch
import numpy as np

from safetensors.torch import save_file as safe_save_file
from transformers.models.wav2vec2.modeling_wav2vec2 import WAV2VEC2_ADAPTER_SAFE_FILE
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset, Audio, Dataset
from evaluate import load

from .args import Options

wer_metric = load("wer")


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

    def filter_short(batch):
        audio = batch["audio"]
        duration = len(audio["array"]) / audio["sampling_rate"]
        return duration > 0.5

    def prepare_dataset(batch):
        audio = batch["audio"]

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

    processor = Wav2Vec2Processor.from_pretrained(
        options.pretrained_model_id,
    )
    processor.tokenizer = Wav2Vec2CTCTokenizer(
        options.vocab_json,
        unk_token=options.unk,
        pad_token=options.pad,
        word_delimiter_token=options.word_delimiter,
        target_lang=options.lang,
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        options.pretrained_model_id,
        target_lang=options.pretrained_model_lang,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
    )

    dev = load_partition(options, "dev", processor)
    train = load_partition(options, "train", processor)

    model.init_adapter_layers()
    model.freeze_base_model()
    adapter_weights = model._get_adapters()
    for param in adapter_weights.values():
        param.requires_grad = True

    training_args = TrainingArguments(
        output_dir=options.model_dir,
        group_by_length=True,
        per_device_train_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=1,
        gradient_checkpointing=True,
        fp16=True,
        learning_rate=1e-3,
        warmup_steps=100,
        save_total_limit=2,
        push_to_hub=False,
        load_best_model_at_end=True,
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

    try:
        trainer.train(resume_from_checkpoint=True)
    except ValueError:  # no checkpoint
        trainer.train()

    adapter_file = WAV2VEC2_ADAPTER_SAFE_FILE.format(model.target_lang)
    adapter_file = options.model_dir / adapter_file

    safe_save_file(model._get_adapters(), adapter_file, metadata={"format": "pt"})
