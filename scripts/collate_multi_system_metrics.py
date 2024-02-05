import json
import os
import pandas as pd
import argparse


def process_pair(devtest_dir, pair, systems):
    if "-" in pair:
        src, tgt = pair.split("-")
        system_lang_scores = {}

        for system in systems:
            try:
                with open(
                    os.path.join(
                        devtest_dir, pair, f"score.{system}.{pair}.json"
                    ),
                    "r",
                ) as f:
                    scores = json.load(f)
                    system_lang_scores[system] = scores[1]["score"]
            except FileNotFoundError:
                system_lang_scores[system] = -1.0
            except Exception as e:
                print(f"Error processing {devtest_dir}/{pair} for {system}: {e}")

        language = src if src != "eng_Latn" else tgt
        return language, system_lang_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--devtest_dir", help="Directory containing devtest data")
    parser.add_argument(
        "--systems",
        help="Comma separated list of systems for which scores are to be collated",
    )
    args = parser.parse_args()

    devtest_dir = args.devtest_dir
    systems = args.systems.split(",")

    all_scores = {}

    pairs = [
        d
        for d in os.listdir(devtest_dir)
        if os.path.isdir(os.path.join(devtest_dir, d))
    ]

    for pair in pairs:
        lang, scores = process_pair(devtest_dir, pair, systems)
        all_scores[lang] = scores

    df = pd.DataFrame.from_dict(all_scores, orient="index")
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Language"}, inplace=True)

    df = df.sort_values(by=["Language"], ascending=True)
    df.to_csv(os.path.join(devtest_dir, "scores.csv"), index=False)
