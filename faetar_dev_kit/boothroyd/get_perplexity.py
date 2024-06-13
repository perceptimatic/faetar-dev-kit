import kenlm
import os
import sys

# for n gram character based language models use trn_lstm files

lm_file_path = sys.argv[1]
trn_file = sys.argv[2]
out_folder = sys.argv[3]
partition = sys.argv[4]

model = kenlm.Model(lm_file_path)

if not os.path.exists(f"{out_folder}/{partition}"):
    os.makedirs(f"{out_folder}/{partition}")

out_path = f"{out_folder}/{partition}/perplexity_{os.path.basename(lm_file_path)}"
out = open(out_path, "w")

with open(trn_file, "r") as f:
    for line in f:
        sentence, file = line.rsplit(maxsplit=1)
        perp = model.perplexity(sentence)
        out.write(f"{perp:.1f}\t{sentence}\t{file}\n")
out.close()
f.close()
