import os
import sys
import string
from tqdm import tqdm
from collections import defaultdict
from typing import List, Tuple, Dict


def read_lines(fname: str) -> List[str]:
    """
    Reads all lines from an input file and returns them as a list of strings.

    Args:
        fname (str): path to the input file to read

    Returns:
        List[str]: a list of strings, where each string is a line from the file
            and returns an empty list if the file does not exist.
    """
    # if path doesnt exist, return empty list
    if not os.path.exists(fname):
        return []

    with open(fname, "r") as f:
        lines = f.readlines()
    return lines


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


def pair_dedup_lists(src_list: List[str], tgt_list: List[str]) -> Tuple[List[str], List[str]]:
    """
    Removes duplicates from two lists by pairing their elements and removing duplicates from the pairs.

    Args:
        src_list (List[str]): a list of strings from source language data.
        tgt_list (List[str]): a list of strings from target language data.

    Returns:
        Tuple[List[str], List[str]]: a tuple of deduplicated version of "`(src_list, tgt_list)`".
    """
    src_tgt = list(set(zip(src_list, tgt_list)))
    src_deduped, tgt_deduped = zip(*src_tgt)
    return src_deduped, tgt_deduped


def pair_dedup_files(src_file: str, tgt_file: str):
    """
    Removes duplicates from two files by pairing their lines and removing duplicates from the pairs.

    Args:
        src_file (str): path to the source language file to deduplicate.
        tgt_file (str): path to the target language file to deduplicate.
    """
    src_lines = read_lines(src_file)
    tgt_lines = read_lines(tgt_file)
    len_before = len(src_lines)

    src_dedupped, tgt_dedupped = pair_dedup_lists(src_lines, tgt_lines)

    len_after = len(src_dedupped)
    num_duplicates = len_before - len_after

    print(f"Dropped duplicate pairs in {src_file} Num duplicates -> {num_duplicates}")
    create_txt(src_file, src_dedupped)
    create_txt(tgt_file, tgt_dedupped)


def strip_and_normalize(line: str) -> str:
    """
    Strips and normalizes a string by lowercasing it, removing spaces and punctuation.

    Args:
        line (str): string to strip and normalize.

    Returns:
        str: stripped and normalized version of the input string.
    """
    # lowercase line, remove spaces and strip punctuation

    # one of the fastest way to add an exclusion list and remove that
    # list of characters from a string
    # https://towardsdatascience.com/how-to-efficiently-remove-punctuations-from-a-string-899ad4a059fb
    exclist = string.punctuation + "\u0964"
    table_ = str.maketrans("", "", exclist)

    line = line.replace(" ", "").lower()
    # dont use this method, it is painfully slow
    # line = "".join([i for i in line if i not in string.punctuation])
    line = line.translate(table_)
    return line


def expand_tupled_list(list_of_tuples: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
    """
    Expands a list of tuples into two lists by extracting the first and second elements of the tuples.

    Args:
        list_of_tuples (List[Tuple[str, str]]): a list of tuples, where each tuple contains two strings.

    Returns:
        Tuple[List[str], List[str]]: a tuple containing two lists, the first being the first elements of the
            tuples in `list_of_tuples` and the second being the second elements.
    """
    # convert list of tuples into two lists
    # https://stackoverflow.com/questions/8081545/how-to-convert-list-of-tuples-to-multiple-lists
    # [(en, as), (as, bn), (bn, gu)] - > [en, as, bn], [as, bn, gu]
    list_a, list_b = map(list, zip(*list_of_tuples))
    return list_a, list_b


def normalize_and_gather_all_benchmarks(devtest_dir: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Normalizes and gathers all benchmark datasets from a directory into a dictionary.

    Args:
        devtest_dir (str): path to the directory containing the subdirectories named after the benchmark datasets, \
            where each subdirectory is named in the format "`src_lang-tgt_lang`" and contain four files: `dev.src_lang`, \
            `dev.tgt_lang`, `test.src_lang`, and `test.tgt_lang` representing the development and test sets for the language pair.

    Returns:
        Dict[str, Dict[str, List[str]]]: a dictionary mapping language pairs (in the format "`src_lang-tgt_lang`") \
            to dictionaries containing two lists, the first being the normalized source language lines and the \
            second being the normalized target language lines for all benchmark datasets.
    """
    devtest_pairs_normalized = defaultdict(lambda: defaultdict(list))

    for benchmark in os.listdir(devtest_dir):
        print(f"{devtest_dir}/{benchmark}")
        for pair in tqdm(os.listdir(f"{devtest_dir}/{benchmark}")):
            src_lang, tgt_lang = pair.split("-")

            src_dev = read_lines(f"{devtest_dir}/{benchmark}/{pair}/dev.{src_lang}")
            tgt_dev = read_lines(f"{devtest_dir}/{benchmark}/{pair}/dev.{tgt_lang}")
            src_test = read_lines(f"{devtest_dir}/{benchmark}/{pair}/test.{src_lang}")
            tgt_test = read_lines(f"{devtest_dir}/{benchmark}/{pair}/test.{tgt_lang}")

            # if the tgt_pair data doesnt exist for a particular test set,
            # it will be an empty list
            if tgt_test == [] or tgt_dev == []:
                print(f"{benchmark} does not have {src_lang}-{tgt_lang} data")
                continue

            # combine both dev and test sets into one
            src_devtest = src_dev + src_test
            tgt_devtest = tgt_dev + tgt_test

            src_devtest = [strip_and_normalize(line) for line in src_devtest]
            tgt_devtest = [strip_and_normalize(line) for line in tgt_devtest]

            devtest_pairs_normalized[pair]["src"].extend(src_devtest)
            devtest_pairs_normalized[pair]["tgt"].extend(tgt_devtest)

    # dedup merged benchmark datasets
    for pair in devtest_pairs_normalized:
        src_devtest = devtest_pairs_normalized[pair]["src"]
        tgt_devtest = devtest_pairs_normalized[pair]["tgt"]

        src_devtest, tgt_devtest = pair_dedup_lists(src_devtest, tgt_devtest)
        devtest_pairs_normalized[pair]["src"] = src_devtest
        devtest_pairs_normalized[pair]["tgt"] = tgt_devtest

    return devtest_pairs_normalized


def remove_train_devtest_overlaps(train_dir: str, devtest_dir: str):
    """
    Removes overlapping data between the training and dev/test (benchmark)
    datasets for all language pairs.

    Args:
        train_dir (str): path of the directory containing the training data.
        devtest_dir (str): path of the directory containing the dev/test data.
    """
    devtest_pairs_normalized = normalize_and_gather_all_benchmarks(devtest_dir)

    all_src_sentences_normalized = []
    for key in devtest_pairs_normalized:
        all_src_sentences_normalized.extend(devtest_pairs_normalized[key]["src"])
    # remove duplicates in all test benchmarks across all lang pair
    # this might not be the most optimal way but this is a tradeoff for generalizing the code at the moment
    all_src_sentences_normalized = list(set(all_src_sentences_normalized))

    src_overlaps = []
    tgt_overlaps = []

    pairs = os.listdir(train_dir)
    for pair in pairs:
        src_lang, tgt_lang = pair.split("-")

        new_src_train, new_tgt_train = [], []

        src_train = read_lines(f"{train_dir}/{pair}/train.{src_lang}")
        tgt_train = read_lines(f"{train_dir}/{pair}/train.{tgt_lang}")

        len_before = len(src_train)
        if len_before == 0:
            continue

        src_train_normalized = [strip_and_normalize(line) for line in src_train]
        tgt_train_normalized = [strip_and_normalize(line) for line in tgt_train]

        src_devtest_normalized = all_src_sentences_normalized
        tgt_devtest_normalized = devtest_pairs_normalized[pair]["tgt"]

        # compute all src and tgt super strict overlaps for a lang pair
        overlaps = set(src_train_normalized) & set(src_devtest_normalized)
        src_overlaps.extend(list(overlaps))

        overlaps = set(tgt_train_normalized) & set(tgt_devtest_normalized)
        tgt_overlaps.extend(list(overlaps))

        # dictionaries offer O(1) lookup
        src_overlaps_dict, tgt_overlaps_dict = {}, {}
        for line in src_overlaps:
            src_overlaps_dict[line] = 1
        for line in tgt_overlaps:
            tgt_overlaps_dict[line] = 1

        # loop to remove the ovelapped data
        idx = 0
        for src_line_norm, tgt_line_norm in tqdm(
            zip(src_train_normalized, tgt_train_normalized), total=len_before
        ):
            if src_overlaps_dict.get(src_line_norm, None):
                continue
            if tgt_overlaps_dict.get(tgt_line_norm, None):
                continue

            new_src_train.append(src_train[idx])
            new_tgt_train.append(tgt_train[idx])
            idx += 1

        len_after = len(new_src_train)
        print(
            f"Detected overlaps between train and devetest for {pair} is {len_before - len_after}"
        )
        print(f"saving new files at {train_dir}/{pair}/")
        create_txt(f"{train_dir}/{pair}/train.{src_lang}", new_src_train)
        create_txt(f"{train_dir}/{pair}/train.{tgt_lang}", new_tgt_train)


if __name__ == "__main__":

    train_data_dir = sys.argv[1]
    # benchmarks directory should contains all the test sets
    devtest_data_dir = sys.argv[2]

    remove_train_devtest_overlaps(train_data_dir, devtest_data_dir)
