import argparse
import os

from datasets import Dataset
import wandb
from pararel_utils import (
    MPARAREL_FOLDER,
    PATTERNS_FOLDER,
    SUBJECT_KEY,
    SUBJECT_QCODE,
    TUPLES_FOLDER,
    get_mpararel_subject_object,
    get_mpararel_templates,
)
from tqdm import tqdm


def main(args):
    dataset = []
    patterns_path = os.path.join(MPARAREL_FOLDER, PATTERNS_FOLDER)
    for lang in tqdm(os.listdir(patterns_path), desc="Languages"):
        os.makedirs(os.path.join(patterns_path, lang), exist_ok=True)
        for relation_filename in tqdm(
            os.listdir(os.path.join(patterns_path, lang)), desc="Relations"
        ):
            templates = get_mpararel_templates(lang, relation_filename)
            tuples_data = get_mpararel_subject_object(
                lang,
                relation_filename,
                tuples_folder=TUPLES_FOLDER + "_with_aliases",
                only_tuple=False,
            )
            relation = relation_filename.replace(".jsonl", "")
            for tuple_ in tuples_data:
                dataset.extend(
                    [
                        {
                            "id": f"{lang}_{relation}_{tuple_[SUBJECT_QCODE]}_{template_id}",
                            "language": lang,
                            "relation": relation,
                            "template": templates[template_id],
                            "template_id": template_id,
                            "query": templates[template_id].replace(
                                "[X]", tuple_[SUBJECT_KEY]
                            ),
                            **tuple_,
                        }
                        for template_id in range(len(templates))
                    ]
                )
    Dataset.from_list(dataset).push_to_hub(args.hf_dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_dataset_name", type=str, default="coastalcph/mpararel_with_aliases"
    )
    args = parser.parse_args()

    wandb.init(project="push_mpararel_hf")

    main(args)
