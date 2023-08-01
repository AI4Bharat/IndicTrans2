import sys


def clean_vocab():
    """
    Cleans a vocabulary file by filtering out invalid lines.
    """
    for line in sys.stdin:
        fields = line.strip("\r\n ").split(" ")
        if len(fields) == 2:
            print(line)


if __name__ == "__main__":
    clean_vocab()
