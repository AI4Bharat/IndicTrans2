from typing import Tuple

import re
import sys
from tqdm import tqdm
from joblib import Parallel, delayed
from .indic_num_map import INDIC_NUM_MAP


URL_PATTERN = r'\b(?<![\w/.])(?:(?:https?|ftp)://)?(?:(?:[\w-]+\.)+(?!\.))(?:[\w/\-?#&=%.]+)+(?!\.\w+)\b'
EMAIL_PATTERN = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}'
# handles dates, time, percentages, proportion, ratio, etc
NUMERAL_PATTERN = r"(~?\d+\.?\d*\s?%?\s?-?\s?~?\d+\.?\d*\s?%|~?\d+%|\d+[-\/.,:']\d+[-\/.,:'+]\d+(?:\.\d+)?|\d+[-\/.:'+]\d+(?:\.\d+)?)"
# handles upi, social media handles and hashtags
OTHER_PATTERN = r'[A-Za-z0-9]*[#|@]\w+'


def normalize_indic_numerals(line: str):
    """
    Normalize the numerals in Indic languages from native script to Roman script (if present).
    
    Args:
        line (str): an input string with Indic numerals to be normalized.
    
    Returns:
        str: an input string with the all Indic numerals normalized to Roman script.
    """
    return "".join([INDIC_NUM_MAP.get(c, c) for c in line])


def wrap_with_placeholders(text: str, patterns: list) -> Tuple[str, dict]:
    """
    Wraps substrings with matched patterns in the given text with placeholders and returns
    the modified text along with a mapping of the placeholders to their original value.
    
    Args:
        text (str): an input string which needs to be wrapped with the placeholders.
        pattern (list): list of patterns to search for in the input string.
    
    Returns:
        Tuple[str, dict]: a tuple containing the modified text and a dictionary mapping 
            placeholders to their original values.
    """
    serial_no = 1
    
    placeholder_entity_map = dict()
    
    for pattern in patterns:
        matches = set(re.findall(pattern, text))
        
        # wrap common match with placeholder tags
        for match in matches:
            if pattern==URL_PATTERN :
                #Avoids false positive URL matches for names with initials.
                temp = match.replace(".",'')
                if len(temp)<4:
                    continue
            if pattern==NUMERAL_PATTERN :
                #Short numeral patterns do not need placeholder based handling.
                temp = match.replace(" ",'').replace(".",'').replace(":",'')
                if len(temp)<4:
                    continue
            
            #Set of Translations of "ID" in all the suppported languages have been collated.            
            #This has been added to deal with edge cases where placeholders might get translated. 
            indic_failure_cases = ['آی ڈی ', 'ꯑꯥꯏꯗꯤ', 'आईडी', 'आई . डी . ', 'ऐटि', 'آئی ڈی ', 'ᱟᱭᱰᱤ ᱾', 'आयडी', 'ऐडि', 'आइडि']         
            placeholder = "<ID{}>".format(serial_no)
            alternate_placeholder = "< ID{} >".format(serial_no)                    
            placeholder_entity_map[placeholder] = match
            placeholder_entity_map[alternate_placeholder] = match
            
            for i in indic_failure_cases:
                placeholder_temp = "<{}{}>".format(i,serial_no)
                placeholder_entity_map[placeholder_temp] = match
                placeholder_temp = "< {}{} >".format(i, serial_no)
                placeholder_entity_map[placeholder_temp] = match
                placeholder_temp = "< {} {} >".format(i, serial_no)
                placeholder_entity_map[placeholder_temp] = match
            
            text = text.replace(match, placeholder)
            serial_no+=1
    
    text = re.sub("\s+", " ", text)
    
    #Regex has failure cases in trailing "/" in URLs, so this is a workaround. 
    text = text.replace(">/",">")
        
    return text, placeholder_entity_map


def normalize(text: str, patterns: list = [EMAIL_PATTERN, URL_PATTERN, NUMERAL_PATTERN, OTHER_PATTERN]) -> Tuple[str, dict]:
    """
    Normalizes and wraps the spans of input string with placeholder tags. It first normalizes
    the Indic numerals in the input string to Roman script. Later, it uses the input string with normalized
    Indic numerals to wrap the spans of text matching the pattern with placeholder tags.
    
    Args:
        text (str): input string.
        pattern (list): list of patterns to search for in the input string.
    
    Returns:
        Tuple[str, dict]: a tuple containing the modified text and a dictionary mapping 
            placeholders to their original values.
    """
    text = normalize_indic_numerals(text.strip("\n"))
    text, placeholder_entity_map  = wrap_with_placeholders(text, patterns)
    return text, placeholder_entity_map
