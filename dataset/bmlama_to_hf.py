import argparse
import os
import collections
import pandas as pd
import wandb
from constants import LANGUAGES
from datasets import Dataset
from tqdm import tqdm
from mpararel_to_hf import filter_trivial_examples


def main(args):
    data_dir = os.path.join(args.bmlama_clone_folder, "1_easyrun/BMLAMA17")
    dataset = []
    ids_to_remove = set()
    metrics = collections.defaultdict(int)
    for lang in tqdm(LANGUAGES, desc="Languages"):
        df = pd.read_csv(os.path.join(data_dir, lang + ".tsv"), sep="\t")
        for index, row in df.iterrows():
            if not row["Prompt"].endswith("<mask>."):
                ids_to_remove.add(index)
                metrics[f"non_autoregressive/{lang}"] += 1
    for lang in tqdm(LANGUAGES, desc="Languages"):
        df = pd.read_csv(os.path.join(data_dir, lang + ".tsv"), sep="\t")
        for index, row in df.iterrows():
            if index in ids_to_remove:
                continue
            dataset.append(
                {
                    "id": f"{lang}_{index}",
                    "language": lang,
                    "query": row["Prompt"].replace("<mask>.", ""),
                    "obj_label": row["Ans"],
                    "sub_label": row["Subject"],
                    "candidates": row["Candidate Ans"],
                }
            )
    wandb.log(metrics)
    ds = Dataset.from_list(dataset)
    ds = filter_trivial_examples(ds)
    ds.push_to_hub(args.hf_dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bmlama_clone_folder", type=str)
    parser.add_argument("--hf_dataset_name", type=str, default="coastalcph/bmlama10")
    args = parser.parse_args()

    wandb.init(project="push_hf_xlingual_ds", name="blama10")

    main(args)
