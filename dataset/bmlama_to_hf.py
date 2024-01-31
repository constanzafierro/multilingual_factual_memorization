import argparse
import os

import pandas as pd
import wandb
from constants import LANGUAGES
from datasets import Dataset
from tqdm import tqdm


def main(args):
    data_dir = os.path.join(args.bmlama_clone_folder, "1_easyrun/BMLAMA17")
    dataset = []
    for lang in tqdm(LANGUAGES, desc=""):
        df = pd.read_csv(os.path.join(data_dir, lang + ".tsv"), sep="\t")
        for index, row in df.iterrows():
            assert row["Prompt"].endswith("<mask>.")
            dataset.append(
                {
                    "id": f"{lang}_{index}",
                    "language": lang,
                    "query": row["Prompt"].replace("<mask>.", ""),
                    "obj_label": row["Ans"],
                    "sub_label": row["Subject"],
                    "candidates": row["Candidate"],
                }
            )
    ds = Dataset.from_list(dataset)
    ds.push_to_hub(args.hf_dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bmlama_clone_folder", type=str)
    parser.add_argument("--hf_dataset_name", type=str, default="coastalcph/bmlama10")
    args = parser.parse_args()

    wandb.init(project="push_mpararel_hf")

    main(args)
