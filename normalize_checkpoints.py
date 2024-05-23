import torch
import os
from sys import argv

# This script will recursively dive into a directory
# and rename all fairseq checkpoints to the legacy 'transformer'
# This will void the need for a custom model_configs file


def main(path):
    for fname in os.listdir(path):
        temp_path = os.path.join(path, fname)
        if os.path.isdir(temp_path):
            main(temp_path)
        elif fname.endswith(".pt"):
            ckpt = torch.load(temp_path)
            if ckpt["cfg"]["model"].arch.startswith("transformer"):
                print(f"normalizing {temp_path}")
                ckpt["cfg"]["model"].arch = "transformer"
                ckpt["cfg"]["model"]._name = "transformer"
                torch.save(ckpt, temp_path)


if __name__ == "__main__":
    main(argv[1])
