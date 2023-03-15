import sys


def clean_vocab(in_vocab_fname: str, out_vocab_fname: str):
    """
    Cleans a vocabulary file by filtering out invalid lines.

    Args:
        in_vocab_fname (str): path of the input vocabulary file.
        out_vocab_fname (str): path of the input vocabulary file.
    """
    with open(in_vocab_fname, "r", encoding="utf-8") as infile, open(
        out_vocab_fname, "w", encoding="utf-8"
    ) as outfile:
        for i, line in enumerate(infile):
            fields = line.strip("\r\n ").split(" ")
            if len(fields) == 2:
                outfile.write(line)
            if len(fields) != 2:
                print(f"{i}: {line.strip()}")
                for c in line:
                    print(f"{c}:{hex(ord(c))}")


if __name__ == "__main__":
    in_vocab_fname = sys.argv[1]
    out_vocab_fname = sys.argv[2]
    clean_vocab(in_vocab_fname, out_vocab_fname)
