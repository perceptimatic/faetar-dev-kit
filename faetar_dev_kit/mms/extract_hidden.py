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
    hidden_dir = Path(hidden_dir).resolve()
    model_dir = Path(model_dir).resolve()
    data_dir = Path(data_dir).resolve()

    model = Wav2Vec2ForCTC.from_pretrained(model_dir, target_lang=lang,
                                           output_hidden_states=True,
                                           ignore_mismatched_sizes=True)

    processor = Wav2Vec2Processor.from_pretrained(model_dir, target_lang=lang,
                                                  ignore_mismatched_sizes=True)

    sampling_rate=processor.feature_extractor.sampling_rate

    print("Loading dataset")
    data = load_dataset("audiofolder", data_dir=data_dir, split="all")
    print("Casting dataset to audio")
    data = data.cast_column(
        "audio", Audio(sampling_rate=sampling_rate)
    )

    os.makedirs(hidden_dir, exist_ok=True)
    states = None

    for audio in tqdm(data["audio"]):
        path = hidden_dir / os.path.splitext(Path(audio["path"]).relative_to(data_dir).as_posix())[0]

        if os.path.exists(f"{path}_48.pt"):
            continue

        inputs = processor(
            audio["array"],
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        states = model(inputs.input_values).hidden_states

        for i in [12, 24, 36, 46, 47, 48]:
            pt = f"{path}_{i}.pt"
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            torch.save(states[i], pt)

    return states

if __name__ == "__main__":
    _, hidden, model, data, lang, *_ = sys.argv
    extract_hidden(hidden, model, data, lang)

