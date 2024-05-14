# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import torch

from pathlib import Path
from datasets import load_dataset, Audio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from tqdm import tqdm

def extract_hidden(hidden_dir, model_dir, data_dir, lang):
    model = Wav2Vec2ForCTC.from_pretrained(model_dir, target_lang=lang,
                                           ignore_mismatched_sizes=True)

    processor = Wav2Vec2Processor.from_pretrained(model_dir, target_lang=lang,
                                                  ignore_mismatched_sizes=True)

    data = load_dataset("audiofolder", data_dir=data_dir, split="all")
    data = data.cast_column(
        "audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate)
    )

    os.makedirs(hidden_dir, exist_ok=True)

    for audio in tqdm(data["audio"]):
        path = os.path.splitext(Path(audio["path"]).relative_to(data_dir).as_posix())[0]

        inputs = processor(
            audio,
            sampling_rate=processor.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
        )

        for i, hidden in enumerate(model(inputs.input_values).hidden_states):
            pt = hidden_dir / (path + f"_{i}.pt")

            torch.save(hidden, pt)

    return 0

if __name__ == "__main__":
    _, hidden, model, data, lang, *_ = sys.argv
    extract_hidden(hidden, model, data, lang)

