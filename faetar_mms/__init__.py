from typing import Optional, Sequence
from .args import Options


def main(args: Optional[Sequence[str]] = None) -> None:
    options = Options.parse_args(args)

    if options.cmd == "write-vocab":
        from .text import write_vocab

        write_vocab(options)
    elif options.cmd == "train":
        from .train import train

        train(options)
    elif options.cmd == "decode":
        from .decode import decode

        decode(options)
    elif options.cmd == "evaluate":
        from .text import evaluate

        evaluate(options)
    elif options.cmd == "metadata-to-trn":
        from .text import metadata_to_trn

        metadata_to_trn(options)
    elif options.cmd == "vocab-to-token2id":
        from .text import vocab_to_token2id

        vocab_to_token2id(options)
    else:
        raise NotImplementedError
