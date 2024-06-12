from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from functools import partial
from textgrids import TextGrid
import pandas as pd
from typing import Generator
from tqdm import tqdm

@dataclass
class Phone:
    onset: float
    offset: float
    phone: str
    speaker: str

    def relativize(self, start: float):
        self.onset -= start
        self.offset -= start

class Alignments:
    data: pd.DataFrame

    def __init__(self, ali_dir: str | Path):
        if isinstance(ali_dir, str): ali_dir = Path(ali_dir)
        bar = tqdm(list(ali_dir.rglob("*.ali")), "Loading Alignments")
        self.data = pd.concat(Alignments.load_ali(ali) for ali in bar)
        print("Extracting Phone objects")
        self.data["object"] = self.data.apply(
            lambda row: Phone(row["onset"], row["offset"],
                              row["phone"], row["speaker"]),
            axis=1
        )
                             
    @staticmethod
    def load_ali(ali: Path):
        """
        Example .ali file:
        onset   offset  speakr phones  "_"
        33.69	33.77	heF001 phones	"t"
        33.77	33.85	heF001 phones	"i"
        33.85	33.90	heF001 phones	"o"
        ...
        """
        cols = ("onset", "offset", "speaker", "phone")
        df = pd.read_csv(ali, sep='\t', names=cols)
        df[["onset", "offset"]] = df[["onset", "offset"]].astype(float)
        df["speaker"] = df["speaker"].apply(lambda s: s.removesuffix(" phones"))
        df["file"] = ali.stem
        return df

    @staticmethod
    def load_textgrid(textgrid: Path):
        """
        Example .ali file:
        ...
        """
        tg = TextGrid(textgrid)
        tier = tg["phones"]

        onset = [p.xmin for p in tier]
        offset = [p.xmax for p in tier]
        phone = [p.text for p in tier]
        speaker = [textgrid.stem for _ in tier]
        
        cols = {"onset": onset, "offset": offset, "speaker": speaker, "phone": phone}
        df = pd.DataFrame(cols)
        df[["onset", "offset"]] = df[["onset", "offset"]].astype(float)
        df["file"] = ali.stem

        return df

class PhoneSeq:
    filename: str
    basename: str
    phones: list[Phone]

    def __init__(self, path: str | Path, alignments: Alignments):
        if isinstance(path, str): path = Path(path)
        self.filename, self.basename = path.name, path.stem
        speaker, start, stop, ali = PhoneSeq.extract_path_info(path)
        data = alignments.data
        self.phones = data[(data["speaker"] == speaker) &
                           (data["file"] == ali) &
                           (data["onset"] >= start) &
                           (data["offset"] <= stop)]["object"].tolist()

        for p in self.phones:
            p.relativize(start)
        
        if len(self.phones) == 0:
            print(f"{self.basename} does not have corresponding entry in .ali")

    @staticmethod
    def extract_path_info(path: Path) -> tuple[str, float, float, str]:
        """ 
        Converts a Path input of a faetar name into its normalized components

        Example file names:
        speaker_start(centiseconds)_stop_recording.extension
        heF003_00000916_00001116_he011.txt
        heF003_00000916_00001116_he011.wav
        """
        speaker, start_str, stop_str, ali = path.stem.split('_')
        start = int(start_str) / 100  # convert to seconds
        stop  = int(stop_str)   / 100
        return speaker, start, stop, ali

    def item_lines(self) -> Generator[str, None, None]:
        """
        EXAMPLE ITEM:

        #file onset offset #phone prev-phone next-phone speaker
        2107 0.3225 0.5225 n ae d 8193
        2107 0.4225 0.5925 d n l 2222
        42 0.4525 0.6525 d n l 2222
        42 0.5225 0.7325 ih l n 8193
        """
        for i in range(1, len(self.phones) - 1):
            file = self.basename
            onset = f"{self.phones[i].onset:.2f}"
            offset = f"{self.phones[i].offset:.2f}"
            phone = self.phones[i].phone
            prev = self.phones[i - 1].phone
            next = self.phones[i + 1].phone
            speaker = self.phones[i].speaker
            yield ' '.join([file, onset, offset, phone, prev, next, speaker])

def make_item(ali_dir: Path, wav_dir: Path) -> Path:
    item_name = f"{wav_dir.name}.item"
    item_path = wav_dir / item_name
    with open(item_path, 'w') as item:
        print(f"Entering {ali_dir}")
        a = Alignments(ali_dir)
        print("#file onset offset #phone prev-phone next-phone speaker",
              file=item)
        print(f"Entering {wav_dir}")
        bar = tqdm(list(wav_dir.rglob("*.wav")), f"Creating {item_name}")
        for w in bar:
            bar.set_postfix_str(w.name)
            p = PhoneSeq(w, a)
            for line in p.item_lines():
                print(line, file=item)
    return item_path

def make_splits(item_path: Path):
    with (open(item_path, 'r') as item,
          open(str(item_path.with_suffix('')) + "_he.item", 'w') as he,
          open(str(item_path.with_suffix('')) + "_hl.item", 'w') as hl):
        head, *body = item.readlines()
        print(head, end='', file=he)
        print(head, end='', file=hl)
        for line in body:
            speaker = line.split()[-1]
            if speaker.startswith('he'):
                print(line, end='', file=he)
            elif speaker.startswith('hl'):
                print(line, end='', file=hl)
            else:
                print("Speaker in line '{line}' does not start with 'he' or 'hl'")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
                    prog='make_item',
                    description='Creates item files from .ali-s and .wavs')

    parser.add_argument('ali_dir', metavar='ali-directory')
    parser.add_argument('wav_dir', metavar='wav-directory')
    parser.add_argument('-s', '--split', action='store_true',
                        help="Create heritage and homeland .item files")

    args = parser.parse_args()

    item = make_item(Path(args.ali_dir), Path(args.wav_dir))
    if args.split:
        make_splits(item)

