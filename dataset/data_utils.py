import os
import string
from functools import partial

import numpy as np
import pandas as pd
import wandb
from datasets import load_dataset, concatenate_datasets

from third_party.rome.util.globals import DATA_DIR
from transformers import (
    LlamaTokenizerFast,
    GPT2TokenizerFast,
    T5TokenizerFast,
    LlamaTokenizer,
    GPTNeoXTokenizerFast,
    XGLMTokenizerFast,
)

TOKENIZER_TO_PREPEND_SPACE = {
    LlamaTokenizerFast: False,
    GPT2TokenizerFast: True,
    T5TokenizerFast: True,
    LlamaTokenizer: False,
    GPTNeoXTokenizerFast: True,
    XGLMTokenizerFast: False,
}


def find_token_range(tokenizer, token_ids_array, subject, prompt):
    if TOKENIZER_TO_PREPEND_SPACE[type(tokenizer)] and not prompt.startswith(subject):
        subject = " " + subject
    subj_tokens = tokenizer.tokenize(subject)
    subj_token_ids = tokenizer.convert_tokens_to_ids(subj_tokens)
    token_ids_array = np.array(token_ids_array.cpu())
    for i in range(len(token_ids_array)):
        if i + len(subj_token_ids) <= len(token_ids_array) and np.all(
            token_ids_array[i : i + len(subj_token_ids)] == subj_token_ids
        ):
            return i, i + len(subj_token_ids)
    if subject[-1] in string.punctuation or subject[-1].isdigit():
        for i in range(len(token_ids_array)):
            if i + len(subj_token_ids) <= len(token_ids_array) and np.all(
                token_ids_array[i : i + len(subj_token_ids) - 1] == subj_token_ids[:-1]
            ):
                return i, i + len(subj_token_ids)
    # If the above failed, we checked the tokens directly as in some languages
    # (e.g. ko) the charachters get paired up differently when in the subject
    # or in the template.
    max_overlap = -1
    overlap_index = -1
    for i in range(len(token_ids_array)):
        if i + len(subj_token_ids) > len(token_ids_array):
            break
        overlap = np.sum(token_ids_array[i : i + len(subj_token_ids)] == subj_token_ids)
        if overlap > max_overlap:
            max_overlap = overlap
            overlap_index = i
    if overlap_index == -1:
        raise Exception(
            "Failed to find subject={} in the token_ids_array={}".format(
                subject, token_ids_array
            )
        )
    remaining_subj = tokenizer.decode(subj_token_ids[max_overlap:])
    last_subj_token = overlap_index + max_overlap + 1
    while last_subj_token <= len(token_ids_array) and remaining_subj.startswith(
        tokenizer.decode(token_ids_array[overlap_index + max_overlap : last_subj_token])
    ):
        last_subj_token += 1
    return overlap_index, last_subj_token


def is_trivial_example(objs, query):
    if not isinstance(objs, list):
        objs = [objs]
    query = query.lower()
    for possible_ans in objs:
        if possible_ans.lower() in query:
            return True
    return False


def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)


def get_start_ans(pred, gt_list):
    start_idx = None
    pred = pred.lower()
    for gt in gt_list:
        id_found = pred.find(gt.lower())
        if id_found != -1 and (start_idx is None or start_idx > id_found):
            start_idx = id_found
    gt_without_punc = [remove_punc(gt) for gt in gt_list]
    if start_idx is None and gt_list != gt_without_punc:
        return get_start_ans(remove_punc(pred), gt_without_punc)
    return start_idx


def add_exact_query(example, memorized_df, df_id_to_index):
    row = memorized_df.iloc[df_id_to_index[example["id"]]]
    start_index = int(row["start_answer"].item())
    if start_index != 0:
        try:
            example["query_inference"] = (
                example["query"].strip() + " " + row["prediction"][:start_index].strip()
            )
        except Exception as e:
            print("example in ds", example["query"])
            print("row", row)
            print("start_index", start_index)
            print("prediction", row["prediction"])
            raise (e)
    else:
        example["query_inference"] = example["query"]
    return example


def log_trivial_examples_counts(memorized_df, ds):
    ds_id_to_index = {ex["id"]: i for i, ex in enumerate(ds)}
    memorized_df["query"] = memorized_df.apply(
        lambda row: ds[ds_id_to_index[row["id"]]]["query"], axis=1
    )
    memorized_df["is_trivial"] = memorized_df.apply(
        lambda row: is_trivial_example(row["ground_truth"], row["query"]), axis=1
    )
    memorized_df["relation"] = memorized_df.apply(
        lambda ex: ex["id"].split("_")[1], axis=1
    )
    memorized_df["template_id"] = memorized_df.apply(
        lambda ex: ex["id"].split("_")[-1], axis=1
    )
    memorized_df["subj_id"] = memorized_df.apply(
        lambda ex: ex["id"].split("_")[-2], axis=1
    )
    memorized_examples = (
        memorized_df[memorized_df.exact_match][
            ["relation", "subj_id", "template_id", "is_trivial"]
        ]
        .groupby(by=["relation", "subj_id", "is_trivial"], as_index=False)
        .count()
    )
    wandb.log(
        {
            # We don't count paraphrases as separate examples.
            "memorized_examples": len(memorized_examples),
            "trivial_memorization_examples": len(
                memorized_examples[memorized_examples.is_trivial]
            ),
            "trivial_memorization": len(memorized_df[memorized_df.is_trivial])
            / len(memorized_df),
        }
    )


def _get_memorized_ds(dataset_name, eval_df_filename):
    inference_df = pd.read_json(eval_df_filename)
    memorized_df = inference_df[inference_df.exact_match].copy()
    ds = load_dataset(dataset_name)["train"]
    ds = ds.filter(lambda ex: ex["id"] in set(memorized_df["id"].values))
    # We might have run inference on more examples than the ones in the ds that we want to use.
    memorized_df = memorized_df[memorized_df["id"].isin(set(ds["id"]))]

    # Check how many trivial.
    if wandb.run is not None:
        log_trivial_examples_counts(memorized_df, ds)

    # Add 'query_inference' with all the tokens before the object.
    memorized_df["start_answer"] = memorized_df.apply(
        lambda ex: get_start_ans(ex["prediction"], ex["ground_truth"]), axis=1
    )
    if len(memorized_df[memorized_df.start_answer.isnull()]) > 0:
        none_values = memorized_df[memorized_df.start_answer.isnull()]
        print(
            "Could not find the answer in the prediction for {} examples. "
            "Data taken from: eval_df_filename={}, dataset_name={}".format(
                len(none_values), eval_df_filename, dataset_name
            )
        )
        if wandb.run is not None:
            wandb.run.summary["answer_not_found"] = len(none_values)
        memorized_df = memorized_df[~memorized_df.start_answer.isnull()]
        ds = ds.filter(lambda ex: ex["id"] in set(memorized_df["id"].values))
    df_id_to_index = {id_: i for i, id_ in enumerate(memorized_df.id.values)}
    ds = ds.map(
        partial(
            add_exact_query, memorized_df=memorized_df, df_id_to_index=df_id_to_index
        )
    )
    return ds


def filter_paraphrases(ds):
    rng = np.random.default_rng(1)
    df = pd.DataFrame(ds)
    relation_subj_to_ids = {
        f"{r}{s}": ids
        for r, s, ids in df[["relation", "sub_uri", "id"]]
        .groupby(by=["relation", "sub_uri"], as_index=False)
        .agg(list)
        .values
    }
    relation_subj_to_chosen_id = {
        r_s: ids[0] if len(ids) == 1 else ids[rng.choice(len(ids), 1)[0]]
        for r_s, ids in relation_subj_to_ids.items()
    }
    return ds.filter(
        lambda ex: ex["id"]
        == relation_subj_to_chosen_id[f"{ex['relation']}{ex['sub_uri']}"]
    )


def get_memorized_dataset(
    dataset_name,
    language,
    eval_dir,
    model_name,
    only_subset,
    filter_trivial=False,
    resample_trivial=False,
    keep_only_trivial=False,
):
    eval_df_filename = os.path.join(
        eval_dir,
        f"{language}--{dataset_name.split('/')[1]}--{model_name}",
        "eval_per_example_records.json",
    )
    if wandb.run is not None:
        wandb.config["eval_df_filename"] = eval_df_filename
    ds = _get_memorized_ds(dataset_name, eval_df_filename)
    ds = filter_paraphrases(ds)
    if only_subset and len(ds) > 1000:
        total = max(1000, int(len(ds) * 0.1))
        rng = np.random.default_rng(0)
        sample_indices = rng.choice(len(ds), total, replace=False)
        ds_sample = ds.select(sample_indices)
        if resample_trivial:
            ds_sample_trivial = ds_sample.filter(
                lambda ex: is_trivial_example(ex["obj_label"], ex["query"])
            )
            sample_indices = set(sample_indices)
            non_trivial_non_selected_ds = ds.filter(
                lambda ex, i: i not in sample_indices
                and not is_trivial_example(ex["obj_label"], ex["query"]),
                with_indices=True,
            )
            extra_sample = non_trivial_non_selected_ds.select(
                rng.choice(
                    len(non_trivial_non_selected_ds),
                    min(len(ds_sample_trivial), len(non_trivial_non_selected_ds)),
                    replace=False,
                )
            )
            ds = concatenate_datasets(
                [
                    ds_sample,
                    extra_sample.filter(
                        lambda ex: not is_trivial_example(ex["obj_label"], ex["query"])
                    ),
                ]
            )
        else:
            ds = ds_sample
    if filter_trivial:
        ds = ds.filter(lambda ex: not is_trivial_example(ex["obj_label"], ex["query"]))
    if keep_only_trivial:
        ds = ds.filter(lambda ex: is_trivial_example(ex["obj_label"], ex["query"]))
    if wandb.run is not None:
        wandb.run.summary["trivial_in_sample"] = len(
            ds.filter(lambda ex: is_trivial_example(ex["obj_label"], ex["query"]))
        )
    return ds
