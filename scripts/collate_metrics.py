import json
import pandas as pd
from multiprocessing import Pool, Manager
from sys import argv
from os import listdir

def process_pair(pair):
    if '-' in pair:
        src, tgt = pair.split('-')

        try:
            with open(f"{devtest_dir}/{pair}/{src}_{tgt}_{system}_scores.txt", 'r') as f:
                scores = json.load(f)
        except FileNotFoundError:
            print(f"Skipping {pair}")
            return ()
        language = src if src != "eng_Latn" else tgt
        return (language, scores[0]['score'], scores[1]['score'])
    else:
        return ()


if __name__ == '__main__':
    devtest_dir = argv[1]
    system = argv[2]

    pairs = listdir(devtest_dir)
    scores_list = Manager().list()

    with Pool() as pool:
        results = pool.map(process_pair, pairs)

    scores_list.extend(results)
    scores_list = list(map(list, scores_list))
    scores = pd.DataFrame(scores_list, columns=['lang', 'bleu', 'chrf2++'])
    scores = scores.sort_values(by=['lang'], ascending=True)
    scores.to_csv(f'{devtest_dir}/{system}_scores.csv', index=False)
