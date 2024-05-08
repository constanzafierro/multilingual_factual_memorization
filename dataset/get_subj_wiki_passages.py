import argparse
import json
import logging
import os
import re
import wandb
import pandas as pd
import requests
from tqdm import tqdm
from dataset.data_utils import get_memorized_dataset

# srnamespace=0 we look only in the main contents of the public pages.
# srqiprofile=classic results are ranked based on "popularity"
SEARCH_KEYWORD = "https://{}.wikipedia.org/w/api.php?action=query&list=search&srnamespace=0&srqiprofile=classic&format=json&srlimit={}&srsearch={}"


def main(args):
    os.makedirs(args.output_folder, exist_ok=True)
    out_filename = os.path.join(args.output_folder, args.language + ".csv")
    if os.path.exists(out_filename):
        print("Existing file.")
        return
    ds = get_memorized_dataset(
        args.dataset_name,
        args.language,
        args.eval_dir,
        args.model_name,
        args.only_subset,
        args.filter_trivial,
        args.resample_trivial,
        args.keep_only_trivial,
    )

    subj_to_passages = {}
    for subject in tqdm(set(ds[["sub_label"]]), desc="Mem. examples"):
        if subject in subj_to_passages:
            continue
        try:
            url = SEARCH_KEYWORD.format(args.language, args.num_paragraphs, subject)
            answer = requests.get(url)
        except Exception as e:
            logging.warning(
                "Failed to fetch results for subject='{}' using url={}".format(
                    subject, url
                )
            )
            logging.warning("Ignored exception: {}".format(e))
        answer = json.loads(answer.content)
        snippets = [item["snippet"] for item in answer["query"]["search"]]
        # The words in the searched word are matched separately.
        subject_regex_match = "({}.*)".format("|".join(subject.split()))
        snippets = [
            re.sub(
                '<span class="searchmatch">{}</span>'.format(
                    re.escape(subject_regex_match)
                ),
                "",
                s,
            ).strip()
            for s in snippets
        ]
        subj_to_passages[subject] = snippets
    pd.DataFrame(
        list(subj_to_passages.items()), columns=["subject", "paragraphs"]
    ).to_csv(out_filename, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_paragraphs", type=int, default=100)
    parser.add_argument("--pararel_folder", type=str, required=True)
    parser.add_argument(
        "--dataset_name",
        required=True,
        type=str,
        help="",
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        help="",
    )
    parser.add_argument(
        "--language",
        type=str,
        help="",
    )
    parser.add_argument("--only_subset", action="store_true")
    parser.add_argument("--filter_trivial", action="store_true")
    parser.add_argument("--keep_only_trivial", action="store_true")
    parser.add_argument("--resample_trivial", action="store_true")
    parser.add_argument("--output_folder", type=str, required=True)
    args = parser.parse_args()

    if "WANDB_NAME" in os.environ:
        run_name = os.getenv("WANDB_NAME")
    wandb.init(project="xfact_retrieve_wiki_passages", name=args.language, config=args)
    args = parser.parse_args()
    main(args)
