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
import warnings

from typing import Any, Union, Optional
from dataclasses import dataclass

import torch
import numpy as np
from torch import nn

from safetensors.torch import save_file as safe_save_file
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    WAV2VEC2_ADAPTER_SAFE_FILE,
    WAV2VEC2_ADAPTER_PT_FILE,
)
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    Wav2Vec2PreTrainedModel,
    TrainingArguments,
    Trainer,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    cached_file,
    is_safetensors_available,
    logging,
    is_torch_greater_or_equal_than_1_13,
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

class Wav2Vec2ForCTCFixed (Wav2Vec2ForCTC):
    def __init__(self, config, target_lang: Optional[str] = None):
        super().__init__(config, target_lang)
    
    def load_adapter(self, target_lang: str, force_load=True, **kwargs):
        r"""
        Load a language adapter model from a pre-trained adapter model.

        Parameters:
            target_lang (`str`):
                Has to be a language id of an existing adapter weight. Adapter weights are stored in the format
                adapter.<lang>.safetensors or adapter.<lang>.bin
            force_load (`bool`, defaults to `True`):
                Whether the weights shall be loaded even if `target_lang` matches `self.target_lang`.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible.
                Will be removed in v5 of Transformers.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.

                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>".

                </Tip>

            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information.

        <Tip>

        Activate the special ["offline-mode"](https://huggingface.co/transformers/installation.html#offline-mode) to
        use this method in a firewalled environment.

        </Tip>

        Examples:

        ```python
        >>> from transformers import Wav2Vec2ForCTC, AutoProcessor

        >>> ckpt = "facebook/mms-1b-all"
        >>> processor = AutoProcessor.from_pretrained(ckpt)
        >>> model = Wav2Vec2ForCTC.from_pretrained(ckpt, target_lang="eng")
        >>> # set specific language
        >>> processor.tokenizer.set_target_lang("spa")
        >>> model.load_adapter("spa")
        ```
        """

        logger = logging.get_logger(__name__)

        if self.config.adapter_attn_dim is None:
            raise ValueError(f"Cannot load_adapter for {target_lang} if `config.adapter_attn_dim` is not defined.")

        if target_lang == self.target_lang and not force_load:
            logger.warning(f"Adapter weights are already set to {target_lang}.")
            return
        
        if is_safetensors_available():
            from safetensors.torch import load_file as safe_load_file

        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", None)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        token = kwargs.pop("token", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        use_safetensors = kwargs.pop("use_safetensors", None if is_safetensors_available() else False)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        model_path_or_id = self.config._name_or_path
        state_dict = None

        # 1. Let's first try loading a safetensors adapter weight
        if use_safetensors is not False:
            filepath = WAV2VEC2_ADAPTER_SAFE_FILE.format(target_lang)

            try:
                weight_path = cached_file(
                    model_path_or_id,
                    filename=filepath,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    cache_dir=cache_dir,
                )

                state_dict = safe_load_file(weight_path)

            except EnvironmentError:
                if use_safetensors:
                    # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                    # to the original exception.
                    raise

            except Exception:
                # For any other exception, we throw a generic error.
                if use_safetensors:
                    raise EnvironmentError(
                        f"Can't load the model for '{model_path_or_id}'. If you were trying to load it"
                        " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                        f" same name. Otherwise, make sure '{model_path_or_id}' is the correct path to a"
                        f" directory containing a file named {filepath}."
                    )

        # 2. If this didn't work let's try loading a PyTorch adapter weight
        if state_dict is None:
            filepath = WAV2VEC2_ADAPTER_PT_FILE.format(target_lang)

            try:
                weight_path = cached_file(
                    model_path_or_id,
                    filename=filepath,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    cache_dir=cache_dir,
                )

                weights_only_kwarg = {"weights_only": True} if is_torch_greater_or_equal_than_1_13 else {}
                state_dict = torch.load(
                    weight_path,
                    map_location="cpu",
                    **weights_only_kwarg,
                )

            except EnvironmentError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                # to the original exception.
                raise

            except Exception:
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load the model for '{model_path_or_id}'. If you were trying to load it"
                    " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{model_path_or_id}' is the correct path to a"
                    f" directory containing a file named {filepath}."
                )

        adapter_weights = self._get_adapters()
        unexpected_keys = set(state_dict.keys()) - set(adapter_weights.keys())
        missing_keys = set(adapter_weights.keys()) - set(state_dict.keys())

        if len(unexpected_keys) > 0:
            raise ValueError(f"The adapter weights {weight_path} has unexpected keys: {', '.join(unexpected_keys)}.")
        elif len(missing_keys) > 0:
            raise ValueError(f"The adapter weights {weight_path} has missing keys: {', '.join(missing_keys)}.")

        # # make sure now vocab size is correct
        # target_vocab_size = state_dict["lm_head.weight"].shape[0]
        # print(self.lm_head)
        # if target_vocab_size != self.config.vocab_size:
        #     self.lm_head = nn.Linear(
        #         self.config.output_hidden_size, target_vocab_size, device=self.device, dtype=self.dtype
        #     )
        #     self.config.vocab_size = target_vocab_size
        #     print(self.lm_head)

        # make sure that adapter weights are put in exactly the same precision and device placement and overwritten adapter weights
        state_dict.pop('lm_head.weight')
        state_dict.pop('lm_head.bias')
        state_dict = {k: v.to(adapter_weights[k]) for k, v in state_dict.items()}
        self.load_state_dict(state_dict, strict=False)

        # set target language corectly
        self.target_lang = target_lang


def train(options: Options):

    processor = Wav2Vec2Processor.from_pretrained(
        options.pretrained_model_id,
    )

    if not options.lang: options.lang = "fae"
    processor.tokenizer = Wav2Vec2CTCTokenizer(
        options.vocab_json,
        unk_token=options.unk,
        pad_token=options.pad,
        word_delimiter_token=options.word_delimiter,
        target_lang=options.lang,
    )

    with open(options.wav2vec2_kwargs_json) as fp:
        wav2vec2_kwargs: dict[str, Any] = json.load(fp)

    for name, expected in (
        ("pad_token_id", processor.tokenizer.pad_token_id),
        ("vocab_size", len(processor.tokenizer)),
        ("target_lang", options.pretrained_model_lang),
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

    model = Wav2Vec2ForCTCFixed.from_pretrained(
            options.pretrained_model_id,
            **wav2vec2_kwargs
    )

    dev = load_partition(options, "dev", processor)
    train = load_partition(options, "train", processor)

    model.init_adapter_layers()
    model.freeze_base_model()
    adapter_weights = model._get_adapters()
    for param in adapter_weights.values():
        param.requires_grad = True

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

    adapter_file = WAV2VEC2_ADAPTER_SAFE_FILE.format(options.lang)
    adapter_file = options.model_dir / adapter_file

    safe_save_file(model._get_adapters(), adapter_file, metadata={"format": "pt"})

    return 0
