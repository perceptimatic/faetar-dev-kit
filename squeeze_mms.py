import sys
import os
import torch

file = sys.argv[1]
out_file = sys.argv[2]

tensor = torch.load(file)

torch.save(torch.squeeze(tensor), out_file)