import os
import sys
from flores_codes_map_indic import flores_to_iso

exp_dir = sys.argv[1]

pairs = os.listdir(exp_dir)
iso_to_flores = {v:k for k, v in flores_to_iso.items()}

for pair in pairs:
    print(pair)
    path = os.path.join(exp_dir, pair)
    src_lang_iso, tgt_lang_iso = pair.split('-')
    
    src_lang = iso_to_flores[src_lang_iso]
    tgt_lang = iso_to_flores[tgt_lang_iso]
    
    for fname in os.listdir(os.path.join(exp_dir, pair)):
        if fname.endswith(src_lang_iso):
            old_fname = os.path.join(path, fname)
            new_fname = os.path.join(path, fname.replace(src_lang_iso, src_lang))
            os.rename(old_fname, new_fname)
        if fname.endswith(tgt_lang_iso):
            old_fname = os.path.join(path, fname)
            new_fname = os.path.join(path, fname.replace(tgt_lang_iso, tgt_lang))
            os.rename(old_fname, new_fname)
    
    new_pair ="{}-{}".format(src_lang, tgt_lang)
    new_path = os.path.join(exp_dir, new_pair)
    os.rename(path, new_path)
