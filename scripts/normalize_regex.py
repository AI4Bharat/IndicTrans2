from typing import Tuple, List

import re
import sys
from tqdm import tqdm
from joblib import Parallel, delayed
from indic_num_map import INDIC_NUM_MAP


URL_PATTERN = r'\b(?<![\w/.])(?:(?:https?|ftp)://)?(?:(?:[\w-]+\.)+(?!\.))(?:[\w/\-?#&=%.]+)+(?!\.\w+)\b'
EMAIL_PATTERN = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}'
# Handles dates, time, percentages, proportion, ratio, etc
NUMERAL_PATTERN = r"(~?\d+\.?\d*\s?%?\s?-?\s?~?\d+\.?\d*\s?%|~?\d+%|\d+[-\/.,:']\d+[-\/.,:'+]\d+(?:\.\d+)?|\d+[-\/.:'+]\d+(?:\.\d+)?)"
# Handles Payment IDs, social media handles and hashtags
OTHER_PATTERN = r'[A-Za-z0-9]*[#|@]\w+'


def normalize_indic_numerals(line: str):
    """
    Normalize the numerals in Indic languages from native script to Roman script (if present).
    
    Args:
        line (str): an input string to be normalized.
    
    Returns:
        str: an input string with the all Indic numerals normalized to Roman script.
    """
    return "".join([INDIC_NUM_MAP.get(c, c) for c in line])


def wrap_with_dnt_tag(src: str, tgt: str, pattern: str) -> Tuple[str, str]:
    """
    Wraps all occurences of a given pattern match that are present in both source and target sentences
    with a do not translate tags (`<dnt>` {input string} `</dnt>`). This will be particularly useful 
    when some span of input string needs to be forwarded as it is and not translated.
    
    Args:
        src (str): source sentence.
        tgt (str): target sentence.
        pattern (str): pattern to search for in the source and target sentence.
    
    Returns:
        Tuple[str, str]: A tuple containing source and target sentences where source sentences 
            are wrapped in `<dnt>` and `</dnt>` tags in case of pattern matches.
    """
    
    # find matches in src and tgt sentence
    src_matches = set(re.findall(pattern, src))
    tgt_matches = set(re.findall(pattern, tgt))
    
    # find matches that are present in both src and tgt
    common_matches = src_matches.intersection(tgt_matches)
    
    # wrap common match with <dnt> and </dnt> tag
    for match in common_matches:
        src = src.replace(match, f' <dnt> {match} </dnt> ')
        tgt = tgt.replace(match, f' <dnt> {match} </dnt> ')
    
    src = re.sub("\s+", " ", src)
    tgt = re.sub("\s+", " ", tgt)
    
    return src, tgt


def normalize(src_line: str, tgt_line: str, patterns: List[str]) -> Tuple[str, str]:
    """
    Normalizes and wraps the spans of text that are present in both source and target sentence 
    with `<dnt>` and `</dnt>` tags. It first normalizes the Indic numerals in the input string to 
    Roman script. Later, it uses the source and target sentence with normalized Indic numerals to 
    wrap the spans of source sentence matching the pattern with `<dnt>` and `</dnt>` tags.
    
    Args:
        src_line (str): source sentence.
        tgt_line (str): source sentence.
        pattern (List[str]): list of patterns to search for in the input string.
    
    Returns:
        Tuple[str, str]: A tuple containing source and target sentences where source sentences 
            are wrapped in `<dnt>` and `</dnt>` tags in case of pattern matches.
    """
    src_line = normalize_indic_numerals(src_line.strip("\n"))
    tgt_line = normalize_indic_numerals(tgt_line.strip("\n"))
    for pattern in patterns:
        src_line, tgt_line = wrap_with_dnt_tag(src_line, tgt_line, pattern)
    return src_line, tgt_line


if __name__ == "__main__":

    src_infname = sys.argv[1]
    tgt_infname = sys.argv[2]
    src_outfname = sys.argv[3]
    tgt_outfname = sys.argv[4]
    
    num_lines = sum(1 for line in open(src_infname, "r"))
    patterns = [EMAIL_PATTERN, URL_PATTERN, NUMERAL_PATTERN, OTHER_PATTERN]

    with open(src_infname, "r", encoding="utf-8") as src_infile, \
        open(tgt_infname, "r", encoding="utf-8") as tgt_infile, \
        open(src_outfname, "w", encoding="utf-8") as src_outfile, \
        open(tgt_outfname, "w", encoding="utf-8") as tgt_outfile:
        
        out_lines = Parallel(n_jobs=-1, backend="multiprocessing")(
            delayed(normalize)(src_line, tgt_line, patterns) for src_line, tgt_line in tqdm(zip(src_infile, tgt_infile), total=num_lines)
        )
        
        for src_line, tgt_line in tqdm(out_lines):
            src_outfile.write(src_line + "\n")
            tgt_outfile.write(tgt_line + "\n")
