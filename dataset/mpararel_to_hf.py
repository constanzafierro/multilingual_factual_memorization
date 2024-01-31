import argparse
import collections
import os

import wandb
from datasets import Dataset
from constants import LANGUAGES
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


def ensure_crosslingual(ds, args):
    ds = ds.filter(lambda ex: ex["language"] in args.languages)
    ds_t0 = ds.filter(lambda ex: ex["template_id"] == 0)

    lang_relation_to_sub = collections.defaultdict(set)
    for ex in ds_t0:
        lang_relation_to_sub[f"{ex['language']}_{ex['relation']}"].add(
            ex[SUBJECT_QCODE]
        )
    relation_to_lang_to_subjs = collections.defaultdict(dict)
    for k, rel_sub in lang_relation_to_sub.items():
        lang, rel = k.split("_")
        relation_to_lang_to_subjs[rel][lang] = rel_sub

    rel_to_shared_subjs = {}
    for rel in relation_to_lang_to_subjs.keys():
        shared_subjs = None
        for lang, subs in relation_to_lang_to_subjs[rel].items():
            if not shared_subjs:
                shared_subjs = subs
            else:
                shared_subjs = shared_subjs.intersection(subs)
        rel_to_shared_subjs[rel] = shared_subjs

    for rel in relation_to_lang_to_subjs.keys():
        print(rel)
        for lang, objs in relation_to_lang_to_subjs[rel].items():
            print(lang, "{:.01%}".format(len(rel_to_shared_subjs[rel]) / len(objs)))

    return ds.filter(
        lambda ex: ex[SUBJECT_QCODE] in rel_to_shared_subjs[ex["relation"]]
    )


def filter_trivial_examples(ds):
    def is_trivial_example(ex):
        objs = ex["obj_label"]
        if not isinstance(objs, list):
            objs = [objs]
        for obj in objs:
            if obj.lower() in ex["query"].lower():
                True
        return False

    filtered = ds.filter(lambda ex: not is_trivial_example(ex))
    if wandb.run is not None:
        wandb.run.summary["trivial_filter/size_before"] = len(ds)
        wandb.run.summary["trivial_filter/size_after"] = len(filtered)
    return filtered


def main(args):
    dataset = []
    patterns_path = os.path.join(MPARAREL_FOLDER, PATTERNS_FOLDER)
    tuples_folder = TUPLES_FOLDER
    if args.use_aliases_folder:
        tuples_folder += "_with_aliases"
    for lang in tqdm(os.listdir(patterns_path), desc="Languages"):
        os.makedirs(os.path.join(patterns_path, lang), exist_ok=True)
        for relation_filename in tqdm(
            os.listdir(os.path.join(patterns_path, lang)), desc="Relations"
        ):
            templates = []
            for template in get_mpararel_templates(lang, relation_filename):
                assert template.endswith("[Y]")
                template = template.replace("[Y]", "").strip()
                templates.append(template)
            tuples_data = get_mpararel_subject_object(
                lang,
                relation_filename,
                tuples_folder=tuples_folder,
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
    ds = Dataset.from_list(dataset)
    ds = filter_trivial_examples(ds)
    ds = ensure_crosslingual(ds, args)
    print("Counts per language")
    print(
        collections.Counter(
            ds.filter(lambda ex: ex["template_id"] == 0)["train"]["language"]
        )
    )
    ds.push_to_hub(args.hf_dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_dataset_name", type=str, default="coastalcph/xlingual_mpararel"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=LANGUAGES,
        help="",
    )
    parser.add_argument("--use_aliases_folder", action="store_true")
    args = parser.parse_args()

    wandb.init(project="push_hf_xlingual_ds", name="mpararel")

    main(args)
