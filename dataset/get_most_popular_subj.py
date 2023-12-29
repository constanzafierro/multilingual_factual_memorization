import argparse
import json
import logging
import os

import requests
from datasets import load_dataset
from pararel_utils import SUBJECT_QCODE
from tqdm import tqdm

URL_SITELINKS_FOR_ENTITY = "https://www.wikidata.org/w/api.php?action=wbgetentities&format=json&ids={}&props=sitelinks/urls"


def get_sitelink_count(qcodes):
    objs_and_count = []
    for qcode in tqdm(qcodes, desc="qcodes"):
        url = URL_SITELINKS_FOR_ENTITY.format(qcode)
        try:
            answer = requests.get(url)
        except Exception as e:
            logging.warning(
                "Failed to fetch sitelinks for entity {} using url={}".format(
                    qcode, url
                )
            )
            logging.warning("Ignored exception: {}".format(e))
        answer = json.loads(answer.content)
        objs_and_count.append((qcode, len(answer["entities"][qcode]["sitelinks"])))
    return sorted(objs_and_count, key=lambda x: x[1], reverse=True)


def main(args):
    ds = load_dataset("coastalcph/xlingual_mpararel")["train"]
    os.makedirs(args.output_dir, exist_ok=True)
    for relation in tqdm(list(set(ds["relation"])), desc="Relations"):
        qcodes_and_counts = get_sitelink_count(
            list(set(ds.filter(lambda ex: ex["relation"] in relation)[SUBJECT_QCODE]))
        )
        with open(os.path.join(args.output_dir, relation + ".json"), "w") as f:
            json.dump(qcodes_and_counts, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="")
    main(parser.parse_args())
