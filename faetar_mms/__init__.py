from typing import Optional, Sequence
from .args import Options


def main(args: Optional[Sequence[str]] = None) -> None:
    options = Options.parse_args(args)

    if options.cmd == "write-vocab":
        from .text import write_vocab

        write_vocab(options)
    else:
        raise NotImplementedError
