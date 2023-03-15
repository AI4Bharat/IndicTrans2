from typing import Tuple

import re
import sys
from tqdm import tqdm
from joblib import Parallel, delayed
from indic_num_map import INDIC_NUM_MAP


URL_PATTERN = r'\b(?<![\w/.])(?:(?:https?|ftp)://)?(?:(?:[\w-]+\.)+(?!\.))(?:[\w/\-?=%.]+)+(?!\.\w+)\b'
EMAIL_PATTERN = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}'
# handles dates, time, percentages, proportion, ratio, etc
NUMERAL_PATTERN = r"(~?\d+\.?\d*\s?%?\s?-?\s?~?\d+\.?\d*\s?%|~?\d+%|\d+[-\/.,:']\d+[-\/.,:'+]\d+(?:\.\d+)?|\d+[-\/.:'+]\d+(?:\.\d+)?)"
# handles upi, social media handles and hashtags
OTHER_PATTERN = r'[A-Za-z0-9]*[#|@]\w+'


def normalize_indic_numerals(line: str):
    return "".join([INDIC_NUM_MAP.get(c, c) for c in line])


def wrap_with_dnt_tag(text: str, pattern: str) -> Tuple[str, str]:
    # find matches in input text
    matches = set(re.findall(pattern, text))
    
    # wrap common match with <dnt> and </dnt> tag
    for match in matches:
        text = text.replace(match, f' <dnt> {match} </dnt> ')
    
    text = re.sub("\s+", " ", text)
    
    return text


def normalize(text, patterns):
    text = normalize_indic_numerals(text.strip("\n"))
    for pattern in patterns:
        text = wrap_with_dnt_tag(text, pattern)
    return text


if __name__ == "__main__":

    src_infname = sys.argv[1]
    src_outfname = sys.argv[2]
    
    num_lines = sum(1 for line in open(src_infname, "r"))
    patterns = [EMAIL_PATTERN, URL_PATTERN, NUMERAL_PATTERN, OTHER_PATTERN]

    with open(src_infname, "r", encoding="utf-8") as src_infile, \
        open(src_outfname, "w", encoding="utf-8") as src_outfile:
        
        for src_line in tqdm(src_infile):
            src_line = normalize(src_line, patterns)
            src_outfile.write(src_line.strip() + "\n")
