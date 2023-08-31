import os
from sys import argv
import multiprocessing as mp


def process_language(lang):

    all_pairs = []
    print(f"lang: {lang}")

    for domain in domains:
        src_fname = f"{base_path}/{domain}/eng_Latn-{lang}/train.eng_Latn"
        tgt_fname = f"{base_path}/{domain}/eng_Latn-{lang}/train.{lang}"

        try:
            with open(src_fname, "r", encoding="utf-8") as f1, open(
                tgt_fname, "r", encoding="utf-8"
            ) as f2:
                src_sents = [x.strip() for x in f1]
                tgt_sents = [x.strip() for x in f2]
            all_pairs.extend([(a, b) for (a, b) in zip(src_sents, tgt_sents)])
        except Exception as e:
            pass

    all_pairs = list(set(all_pairs))
    src_sents, tgt_sents = zip(*all_pairs)

    os.makedirs(f"{out_dir}/eng_Latn-{lang}", exist_ok=True)
    with open(
        f"{out_dir}/eng_Latn-{lang}/train.eng_Latn", "w", encoding="utf-8"
    ) as f1, open(
        f"{out_dir}/eng_Latn-{lang}/train.{lang}", "w", encoding="utf-8"
    ) as f2:
        f1.write("\n".join(src_sents))
        f2.write("\n".join(tgt_sents))


if __name__ == "__main__":

    base_path = argv[1]
    out_dir = argv[2]

    language_codes = [
    'asm_Beng', 'ben_Beng', 'brx_Deva', 'doi_Deva', 'gom_Deva', 
    'guj_Gujr', 'hin_Deva', 'kan_Knda', 'kas_Arab', 'kas_Deva',
    'mai_Deva', 'mal_Mlym', 'mar_Deva', 'mni_Beng', 'mni_Mtei', 
    'npi_Deva', 'ory_Orya', 'pan_Guru', 'san_Deva', 'sat_Olck', 
    'snd_Arab', 'snd_Deva', 'tam_Taml', 'tel_Telu', 'urd_Arab'
    ]

    domains = os.listdir(base_path)

    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(process_language, language_codes)
