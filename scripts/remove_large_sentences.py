import sys
from tqdm import tqdm
from typing import List, Tuple


def remove_large_sentences(src_path: str, tgt_path: str) -> Tuple[int, List[str], List[str]]:
    """
    Removes large sentences from a parallel dataset of source and target data.

    Args:
        src_path (str): path to the file containing the source language data.
        tgt_path (str): path to the file containing the target language data.

    Returns:
        Tuple[int, List[str], List[str]]: a tuple of
            - an integer representing the number of sentences removed
            - a list of strings containing the source language data after removing large sentences
            - a list of strings containing the target language data after removing large sentences
    """
    count = 0
    new_src_lines, new_tgt_lines = [], []

    src_num_lines = sum(1 for line in open(src_path, "r", encoding="utf-8"))
    tgt_num_lines = sum(1 for line in open(tgt_path, "r", encoding="utf-8"))
    assert src_num_lines == tgt_num_lines

    with open(src_path, encoding="utf-8") as f1, open(tgt_path, encoding="utf-8") as f2:
        for src_line, tgt_line in tqdm(zip(f1, f2), total=src_num_lines):
            src_tokens = src_line.strip().split(" ")
            tgt_tokens = tgt_line.strip().split(" ")

            if len(src_tokens) > 200 or len(tgt_tokens) > 200:
                count += 1
                continue

            new_src_lines.append(src_line)
            new_tgt_lines.append(tgt_line)

    return count, new_src_lines, new_tgt_lines


def create_txt(out_file: str, lines: List[str]):
    """
    Creates a text file and writes the given list of lines to file.

    Args:
        out_file (str): path to the output file to be created.
        lines (List[str]): a list of strings to be written to the output file.
    """
    add_newline = not "\n" in lines[0]
    outfile = open("{}".format(out_file), "w", encoding="utf-8")
    for line in lines:
        if add_newline:
            outfile.write(line + "\n")
        else:
            outfile.write(line)
    outfile.close()


if __name__ == "__main__":

    src_path = sys.argv[1]
    tgt_path = sys.argv[2]
    new_src_path = sys.argv[3]
    new_tgt_path = sys.argv[4]

    count, new_src_lines, new_tgt_lines = remove_large_sentences(src_path, tgt_path)
    print(f"{count} lines removed due to seq_len > 200")
    create_txt(new_src_path, new_src_lines)
    create_txt(new_tgt_path, new_tgt_lines)
