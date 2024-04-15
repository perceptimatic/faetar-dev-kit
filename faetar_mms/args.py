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

import argparse
import pathlib
import string

from typing import Literal, Optional, Sequence, Protocol, TypeVar


WS = set(string.whitespace)
T = TypeVar("T")


class ArgparseType(Protocol[T]):
    metavar: str

    @staticmethod
    def to(arg: str) -> T: ...


class StringType(ArgparseType[str]):
    metavar = "STR"

    @staticmethod
    def to(arg: str) -> str:
        return arg


class TokenType(ArgparseType[str]):
    metavar = "TOK"

    @staticmethod
    def to(arg: str) -> str:
        if WS & set(arg):
            raise argparse.ArgumentTypeError(f"'{arg}' contains whitespace")
        return arg


class IntegerType(ArgparseType[int]):
    metavar = "INT"

    @staticmethod
    def to(arg: str) -> int:
        return int(arg)


class NonnegType(ArgparseType[int]):
    metavar = "NONNEG"

    @staticmethod
    def to(arg: str) -> int:
        int_ = int(arg)
        if int_ < 0:
            raise argparse.ArgumentTypeError(f"'{arg}' is not a non-negative integer")
        return int_


class NatType(ArgparseType[int]):
    metavar = "NATG"

    @staticmethod
    def to(arg: str) -> int:
        int_ = int(arg)
        if int_ <= 0:
            raise argparse.ArgumentTypeError(f"'{arg}' is not a natural number")
        return int_


class PathType(ArgparseType[pathlib.Path]):
    metavar = "PTH"

    @staticmethod
    def to(arg: str) -> pathlib.Path:
        return pathlib.Path(arg)


class WriteDirType(PathType):
    metavar = "DIR"


class WriteFileType(PathType):
    metavar = "FILE"


class ReadDirType(ArgparseType[pathlib.Path]):
    metavar = "DIR"

    @staticmethod
    def to(arg: str) -> pathlib.Path:
        pth = pathlib.Path(arg)
        if not pth.is_dir():
            raise argparse.ArgumentTypeError(f"'{arg}' is not a directory")
        return pth


class ReadFileType(ArgparseType[pathlib.Path]):
    metavar = "FILE"

    @staticmethod
    def to(arg: str) -> pathlib.Path:
        pth = pathlib.Path(arg)
        if not pth.is_file():
            raise argparse.ArgumentTypeError(f"'{arg}' is not a file")
        return pth


class Options(object):

    @classmethod
    def _add_argument(
        cls,
        parser: argparse.ArgumentParser,
        *name_or_flags: str,
        type: Optional[type[ArgparseType]] = None,
        help: Optional[str] = None,
    ):
        if type is None:
            type_, metavar = StringType.to, StringType.metavar
        else:
            type_, metavar = type.to, type.metavar
        if name_or_flags[0].startswith("-"):
            default = getattr(cls, name_or_flags[0].lstrip("-").replace("-", "_"))
        else:
            default = None
        parser.add_argument(
            *name_or_flags, metavar=metavar, default=default, type=type_, help=help
        )

    # global kwargs
    unk: str = "[UNK]"
    pad: str = "[PAD]"
    word_delimiter: str = "_"
    sampling_rate: int = 16_000

    # global args
    cmd: Literal["write-vocab", "train", "decode"]

    # write-vocab kwargs
    lang: str = "fae"  # There's no Faetar ISO 639 code, but "fae" isn't mapped yet
    prune_count: int = 0
    append: bool = False

    # write-vocab args
    metadata_csv: pathlib.Path
    vocab_json: pathlib.Path

    # train kwargs
    pretrained_model_id: str = "facebook/mms-1b-all"
    pretrained_model_lang: str = "ita"

    # train args
    # vocab_json
    train_data: pathlib.Path
    dev_data: pathlib.Path
    ckpt_dir: pathlib.Path

    # decode args
    # ckpt_dir
    decode_data: pathlib.Path

    @classmethod
    def add_write_vocab_args(cls, parser: argparse.ArgumentParser):

        cls._add_argument(
            parser, "--prune-count", type=NonnegType, help="Prune tokens <= this count"
        )
        parser.add_argument(
            "--append",
            action="store_true",
            default=False,
            help="Add language to existing file",
        )

        cls._add_argument(
            parser,
            "metadata_csv",
            type=ReadFileType,
            help="Path to training metadata.csv file",
        )
        cls._add_argument(
            parser,
            "vocab_json",
            type=WriteFileType,
            help="Path to vocab.json file (output)",
        )

    @classmethod
    def add_train_args(cls, parser: argparse.ArgumentParser):

        cls._add_argument(
            parser, "--pretrained-model-id", help="model to load from hub"
        )
        cls._add_argument(
            parser, "--pretrained-model-lang", help="iso 639 code of model from hub"
        )

        cls._add_argument(
            parser, "vocab_json", type=ReadFileType, help="Path to vocab.json file"
        )
        cls._add_argument(
            parser, "train_data", type=ReadDirType, help="Path to training AudioFolder"
        )
        cls._add_argument(
            parser, "dev_data", type=ReadDirType, help="Path to development AudioFolder"
        )
        cls._add_argument(
            parser,
            "model_dir",
            type=WriteDirType,
            help="Path to model dir (output)",
        )

    @classmethod
    def add_decode_args(cls, parser: argparse.ArgumentParser):

        cls._add_argument(
            parser,
            "model_dir",
            type=ReadDirType,
            help="Path to model dir",
        )
        cls._add_argument(
            parser,
            "decode_data",
            type=ReadDirType,
            help="Path to AudioFolder to decode",
        )

    @classmethod
    def parse_args(cls, args: Optional[Sequence[str]] = None, **kwargs):
        parser = argparse.ArgumentParser(**kwargs)

        cls._add_argument(
            parser, "--unk", type=TokenType, help="out-of-vocabulary type (string)"
        )
        cls._add_argument(
            parser, "--pad", type=TokenType, help="padding/blank type (string)"
        )
        cls._add_argument(
            parser,
            "--word-delimiter",
            type=TokenType,
            help="word delimiter type (string)",
        )
        cls._add_argument(parser, "--lang", type=TokenType, help="iso 639 code")
        cls._add_argument(
            parser, "--sampling-rate", type=NatType, help="audio sampling rate"
        )

        cmds = parser.add_subparsers(
            dest="cmd", required=True, description="Subcommand (see README)"
        )
        cls.add_write_vocab_args(
            cmds.add_parser("write-vocab", help="Write vocab.json file")
        )
        cls.add_train_args(cmds.add_parser("train", help="fine-tune an mms model"))
        cls.add_decode_args(
            cmds.add_parser("decode", help="decode with fine-tuned mms model")
        )

        return parser.parse_args(args, namespace=cls())
