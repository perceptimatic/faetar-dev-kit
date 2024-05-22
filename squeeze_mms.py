import sys
import os
import torch
import re

file_dir = sys.argv[1]
layer = sys.argv[2]
part = sys.argv[3]

for filename in os.listdir(file_dir):
    if re.search(f'{layer}...$',filename):
        file = os.path.join(file_dir, filename)

        if not os.path.isfile(file):
            continue

        out_path = f"data/mms_hidden/layer_{layer}/{part}/feat/{filename}"

        tensor = torch.load(file)

        torch.save(torch.squeeze(tensor), out_path)

        os.remove(file)