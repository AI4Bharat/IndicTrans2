import os
import sys
from flores_codes_map_indic import flores_to_iso


def convert_iso_to_flores(data_dir: str):
    """
    Converts ISO language code to flores language code for a given directory of language pairs. 
    Assumes that each subdirectory of the given directory corresponds to a language pair, and 
    that each subdirectory contains files named according to the ISO language codes of the source 
    and target languages.

    Args:
        data_dir (str): path of the directory containing the data files for language pairs in ISO language code.
    """
    pairs = os.listdir(data_dir)
    iso_to_flores = {v:k for k, v in flores_to_iso.items()}

    for pair in pairs:
        print(pair)
        path = os.path.join(data_dir, pair)
        src_lang_iso, tgt_lang_iso = pair.split('-')
        
        src_lang = iso_to_flores[src_lang_iso]
        tgt_lang = iso_to_flores[tgt_lang_iso]
        
        for fname in os.listdir(os.path.join(data_dir, pair)):
            if fname.endswith(src_lang_iso):
                old_fname = os.path.join(path, fname)
                new_fname = os.path.join(path, fname.replace(src_lang_iso, src_lang))
                os.rename(old_fname, new_fname)
            
            if fname.endswith(tgt_lang_iso):
                old_fname = os.path.join(path, fname)
                new_fname = os.path.join(path, fname.replace(tgt_lang_iso, tgt_lang))
                os.rename(old_fname, new_fname)
        
        new_pair ="{}-{}".format(src_lang, tgt_lang)
        new_path = os.path.join(data_dir, new_pair)
        os.rename(path, new_path)


if __name__ == "__main__":
    data_dir = sys.argv[1]
    
    convert_iso_to_flores(data_dir)