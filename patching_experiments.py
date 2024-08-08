import argparse
import collections
import json
import os
from itertools import product

import numpy as np
import pandas as pd
import torch
import wandb
from tqdm import tqdm

from dataset.data_utils import find_token_range, get_dataset_name, get_memorized_dataset
from inference.run_inference import prepare_prompt
from model_utils import load_model_and_tok
from patching_utils import trace_important_states, trace_important_window
from third_party.rome.experiments.causal_trace import (
    decode_tokens,
)


def get_token_indices(token_to_patch, examples, input_ids, input_prompts, tokenizer):
    if token_to_patch == "last":
        token_idx_to_patch_from = -1
        token_idx_to_patch = -1
    elif token_to_patch == "last_subject_token":
        subj_ranges = []
        for ex, inp, prompt in zip(examples, input_ids, input_prompts):
            subj_ranges.append(
                find_token_range(tokenizer, inp, ex["sub_label"], prompt)
            )
        token_idx_to_patch_from = subj_ranges[0][-1] - 1
        token_idx_to_patch = subj_ranges[1][-1] - 1
    return token_idx_to_patch_from, token_idx_to_patch


def patch_ex1_into_ex2(mt, ex1, ex2, num_layers, kind, window, token_to_patch="last"):
    "Patch the repr from ex1 into the forward pass of ex2."
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_prompts = []
    for ex in [ex1, ex2]:
        prompt = ex["query_inference"]
        input_prompts.append(
            prepare_prompt(prompt, args.model_name_or_path, "", "t5" in mt.model_name)
        )
    mt.tokenizer.padding_side = "left"
    inp = mt.tokenizer(input_prompts, return_tensors="pt", padding=True).to(device)
    token_idx_to_patch_from, token_idx_to_patch = get_token_indices(
        token_to_patch, [ex1, ex2], inp["input_ids"], input_prompts, mt.tokenizer
    )

    output = mt.model.generate(
        **inp, max_new_tokens=1, return_dict_in_generate=True, output_logits=True
    )
    probs = torch.softmax(output.logits[-1], dim=-1)
    preds_probs, preds_tokens = torch.max(probs, dim=-1)
    base_entropies = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

    answers = decode_tokens(mt.tokenizer, preds_tokens)
    if kind is None:
        results = trace_important_states(
            mt.model,
            num_layers,
            inp,
            e_range=None,
            answer_t=preds_tokens,
            noise=None,
            ntoks=[(token_idx_to_patch_from, token_idx_to_patch)],
        )
    else:
        results = trace_important_window(
            mt.model,
            num_layers,
            inp,
            e_range=None,
            answer_t=preds_tokens,
            kind=kind,
            window=window,
            noise=None,
            ntoks=[(token_idx_to_patch_from, token_idx_to_patch)],
        )
    probs, ranks, ranks_from_tokens, pred_token, entropy = [
        r.detach().cpu() for r in results
    ]
    return dict(
        input_ids=inp["input_ids"].detach().cpu().numpy(),
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"]),
        base_probs=preds_probs,
        base_answer_tokens=preds_tokens,
        base_answers=answers,
        base_entropies=base_entropies,
        window=window,
        # The probability of getting each of the answers.
        patch_probs=probs,
        patch_ranks=ranks,
        patch_ranks_tokens=ranks_from_tokens,
        patch_pred_token=pred_token,
        patch_entropy=entropy,
        patched_tokens_from_to=(token_idx_to_patch_from, token_idx_to_patch),
        kind=kind or "",
    )


def get_xlingual_mem_ids(ds, ds_other):
    df = pd.DataFrame(list(ds) + list(ds_other))
    groupped = (
        df[["relation", "sub_uri", "obj_uri", "language"]]
        .drop_duplicates()
        .groupby(by=["relation", "sub_uri", "obj_uri"], as_index=False)
        .agg(list)
    )
    mem_shared_en = groupped[
        groupped.language.apply(lambda langs: len(langs) > 1)
    ].reset_index()
    langs = [ds[0]["language"], ds_other[0]["language"]]
    return [
        (f"{langs[0]}_{r}_{s}", f"{langs[1]}_{r}_{s}")
        for r, s in mem_shared_en[["relation", "sub_uri"]].values
    ]


def get_last_token_experiment_ids(ds, ds_other):
    source_subj_to_relations = collections.defaultdict(list)
    target_subj_to_relations = collections.defaultdict(list)
    for ex in ds:
        source_subj_to_relations[ex["sub_uri"]].append(ex["relation"])
    for ex in ds_other:
        target_subj_to_relations[ex["sub_uri"]].append(ex["relation"])
    data = []
    for s, relations in source_subj_to_relations.items():
        data.append(
            {
                "sub_uri": s,
                "source_relations": relations,
                "target_relations": target_subj_to_relations[s],
            }
        )
    df = pd.DataFrame(data)
    df["combinations"] = df.apply(
        lambda row: [
            pair
            for pair in product(row["source_relations"], row["target_relations"])
            if pair[0] != pair[1]
        ],
        axis=1,
    )
    subj_relations = df[df.combinations.astype(bool)][
        ["sub_uri", "combinations"]
    ].values
    langs = [ds[0]["language"], ds_other[0]["language"]]
    examples_ids = []
    for s, relation_pairs in subj_relations:
        for r_source, r_target in relation_pairs:
            examples_ids.append(
                (f"{langs[0]}_{r_source}_{s}", f"{langs[1]}_{r_target}_{s}")
            )
    return examples_ids


def main(args):
    data_id = "_".join(
        [
            args.language,
            args.token_to_patch,
        ]
    )
    if args.only_subset:
        data_id = data_id + "_subset"
    if not args.filter_trivial:
        wandb.run.name += " w_trivial"
        data_id += "_w_trivial"
    if args.resample_trivial:
        wandb.run.name += " resample_trivial"
        data_id += "_resample_trivial"
    if args.keep_only_trivial:
        wandb.run.name += " only_trivial"
        data_id += "_only_trivial"
    if args.patch_k_layers != 10:
        data_id += f"_window={args.patch_k_layers}"
    output_folder = os.path.join(args.output_folder, args.model_name, data_id)
    wandb.config["final_output_folder"] = output_folder
    os.makedirs(output_folder, exist_ok=True)

    mt = load_model_and_tok(args.model_name_or_path, args.model_name)

    dataset_name = get_dataset_name(args.model_name, args.language)
    ds = get_memorized_dataset(
        dataset_name,
        args.language,
        args.eval_dir,
        args.model_name,
        args.only_subset,
        mt.tokenizer,
        args.filter_trivial,
        args.resample_trivial,
        args.keep_only_trivial,
    )
    # Note that we are only keeping one template.
    id_to_ex1 = {ex["id"][: ex["id"].rfind("_")]: ex for ex in ds}
    counts = collections.defaultdict(int)
    for lang in tqdm(args.languages_to_patch, desc="Languages"):
        if lang == args.language:
            continue
        dataset_name = get_dataset_name(args.model_name, lang)
        ds_other = get_memorized_dataset(
            dataset_name,
            lang,
            args.eval_dir,
            args.model_name,
            args.only_subset,
            mt.tokenizer,
            args.filter_trivial,
            args.resample_trivial,
            args.keep_only_trivial,
            log_to_wandb=False,
        )
        id_to_ex2 = {ex["id"][: ex["id"].rfind("_")]: ex for ex in ds_other}
        token_to_patch = args.token_to_patch
        if args.token_to_patch == "last_subject_token":
            ids_to_patch = get_xlingual_mem_ids(ds, ds_other)
        elif args.token_to_patch == "last":
            ids_to_patch = get_last_token_experiment_ids(ds, ds_other)
        elif args.token_to_patch == "last_same_ex":
            ids_to_patch = get_xlingual_mem_ids(ds, ds_other)
            token_to_patch = "last"

        counts[f"patched_examples/{lang}"] = len(ids_to_patch)
        for ex_id_source, ex_id_target in tqdm(ids_to_patch, desc="Examples"):
            filename = os.path.join(
                output_folder, f"{ex_id_source}_{ex_id_target}_{args.kind}.npz"
            )
            if not os.path.isfile(filename) or args.override_results:
                result = patch_ex1_into_ex2(
                    mt,
                    id_to_ex1[ex_id_source],
                    id_to_ex2[ex_id_target],
                    mt.num_layers,
                    kind=args.kind,
                    window=args.patch_k_layers,
                    token_to_patch=token_to_patch,
                )
                numpy_result = {
                    k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                    for k, v in result.items()
                }
                result["source_lang"] = args.language
                result["target_lang"] = lang
                result["ex_id_source"] = ex_id_source
                result["ex_id_target"] = ex_id_target
                np.savez(filename, **numpy_result)
    wandb.log({k: c for k, c in counts.items()})

    print("Writing config")
    with open(os.path.join(output_folder, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        required=True,
        type=str,
        help="",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="",
    )
    parser.add_argument(
        "--output_folder",
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
        "--kind",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--language",
        type=str,
        help="",
    )
    parser.add_argument(
        "--languages_to_patch",
        nargs="+",
        help="",
    )
    parser.add_argument("--only_subset", action="store_true")
    parser.add_argument("--filter_trivial", action="store_true")
    parser.add_argument("--keep_only_trivial", action="store_true")
    parser.add_argument("--resample_trivial", action="store_true")
    parser.add_argument("--override_results", action="store_true")
    parser.add_argument("--patch_k_layers", type=int, default=10)
    parser.add_argument(
        "--token_to_patch",
        type=str,
        choices=["last", "last_subject_token", "last_same_ex"],
        default=None,
    )
    args = parser.parse_args()
    if not args.model_name:
        args.model_name = args.model_name_or_path.replace("/", "__")
    wandb.init(
        project=f"patch_{args.token_to_patch}_mpararel",
        name=" ".join([args.model_name, args.language, args.token_to_patch]),
        config=args,
    )
    main(args)
