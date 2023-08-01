import pandas as pd 
from sys import argv
from os import listdir 
from os import makedirs
from tqdm import tqdm
from collections import defaultdict

src_dir = argv[1]
tgt_dir = argv[2]
k = float(argv[3])

total = 0
total_deduped = 0
final_en = []
en_count = {}
final_dict = defaultdict(list)
en_dict = defaultdict(list)


for lang_pair in sorted(listdir(src_dir)):
    lang = lang_pair.split('-')[1]
    with open(f"{src_dir}/{lang_pair}/labse.txt") as fs, \
         open(f"{src_dir}/{lang_pair}/train.eng_Latn") as fe, \
         open(f"{src_dir}/{lang_pair}/train.{lang}", encoding='utf-8') as fl:
        labse = [float(x.strip()) for x in fs]
        en = [x.strip() for x in fe]
        indic = [x.strip() for x in fl]

    df = pd.DataFrame(list(zip(indic, en, labse)), columns =['indic', 'en', 'labse'])
    print(f"{lang} - full data: {len(df)/1e6}M", end=' | ')
    
    df = df.sort_values(by='labse', ascending=False)

    df = df.drop_duplicates(subset='en', keep='first')
    print(f"{lang} - after monolingual dedup: {len(df)/1e6}M", end=' | ')

    # k = threshold
    df = df.loc[df['labse'] > k]
    print(f"{lang} - after HQ filteration: {len(df)/1e6}M (k = {k})", end=' ')

    en = df['en'].tolist()
    indic = df['indic'].tolist()
    en_count[lang] = len(en)
    total += len(en)

    for (en_, indic_) in zip(en, indic):
        en_dict[en_].append((lang, indic_))
    
    print()

print(f"total: {total/1e6}M")

for (en_, translations) in tqdm(en_dict.items()):
    lang, indic_ = min(translations, key=lambda x: en_count[x[0]])
    final_dict[lang].append((en_, indic_))

for (lang, pairs) in final_dict.items():
    en, indic = zip(*pairs)
    total_deduped += len(en)
    print(f"{lang} - after crosslingual dedup: {len(en)/1e6}M")

    makedirs(f"{tgt_dir}/eng_Latn-{lang}", exist_ok=True) 
    with open(f"{tgt_dir}/eng_Latn-{lang}/train.eng_Latn", 'w') as fe, \
         open(f"{tgt_dir}/eng_Latn-{lang}/train.{lang}", 'w', encoding='utf-8') as fl:
        fe.write('\n'.join(en))
        fl.write('\n'.join(indic))

print(f"total_deduped: {total_deduped/1e6}M")
