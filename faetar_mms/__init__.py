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


from typing import Optional, Sequence

from .args import Options


def main(args: Optional[Sequence[str]] = None) -> None:
    options = Options.parse_args(
        args, description="Run a step of the faetar-mms recipe"
    )

    if options.cmd == "compile-metadata":
        from .io import compile_metadata

        compile_metadata(options)
    elif options.cmd == "write-vocab":
        from .io import write_vocab

        write_vocab(options)
    elif options.cmd == "train":
        from .train import train

        train(options)
    elif options.cmd == "decode":
        from .decode import decode

        decode(options)
    elif options.cmd == "evaluate":
        from .io import evaluate

        evaluate(options)
    elif options.cmd == "metadata-to-trn":
        from .io import metadata_to_trn

        metadata_to_trn(options)
    elif options.cmd == "vocab-to-token2id":
        from .io import vocab_to_token2id

        vocab_to_token2id(options)
    else:
        raise NotImplementedError
