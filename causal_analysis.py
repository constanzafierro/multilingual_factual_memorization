import argparse
import collections
import os
import string
from collections import defaultdict
from functools import partial
from glob import glob

import numpy
import numpy as np
import pandas as pd
import torch
import wandb
from accelerate import Accelerator
from datasets import load_dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
)

from third_party.rome.experiments.causal_trace import (
    ModelAndTokenizer,
    collect_embedding_std,
    decode_tokens,
    layername,
    make_inputs,
    plot_trace_heatmap,
    predict_from_input,
    predict_token,
)
from third_party.rome.util import nethook

torch.set_grad_enabled(False)


def load_model_and_tok(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to("cuda")
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=not isinstance(model, LlamaForCausalLM),
    )
    return ModelAndTokenizer(
        model_name=args.model_name, model=model, tokenizer=tokenizer
    )


def trace_with_patch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    trace_layers=None,  # List of traced outputs to return
):
    """Copy of the function in causal_trace.ipynb"""
    prng = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng.randn(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,
        edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**inp)

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced

    return probs


def find_token_range(tokenizer, token_array, subject):
    subj_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(subject))
    token_array = np.array(token_array.cpu())
    for i in range(len(token_array)):
        if i + len(subj_tokens) <= len(token_array) and np.all(
            token_array[i : i + len(subj_tokens)] == subj_tokens
        ):
            return i, i + len(subj_tokens)
    if subject[-1] in string.punctuation:
        return find_token_range(tokenizer, token_array, subject[-1])
    raise Exception(
        "Did not find subj_tokens={} in token_array={}".format(subj_tokens, token_array)
    )


def calculate_hidden_flow(
    mt, prompt, subject, samples=10, noise=0.1, window=10, kind=None
):
    """
    Copy of the function in causal_trace.ipynb
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(mt.model, inp)]
    [answer] = decode_tokens(mt.tokenizer, [answer_t])
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
    low_score = trace_with_patch(
        mt.model, inp, [], answer_t, e_range, noise=noise
    ).item()
    if not kind:
        differences = trace_important_states(
            mt.model, mt.num_layers, inp, e_range, answer_t, noise=noise
        )
    else:
        differences = trace_important_window(
            mt.model,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            window=window,
            kind=kind,
        )
    differences = differences.detach().cpu()
    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        subject_range=e_range,
        answer=answer,
        window=window,
        kind=kind or "",
    )


def trace_important_states(model, num_layers, inp, e_range, answer_t, noise=0.1):
    """Copy of the function in causal_trace.ipynb"""
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, num_layers):
            r = trace_with_patch(
                model,
                inp,
                [(tnum, layername(model, layer))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def trace_important_window(
    model, num_layers, inp, e_range, answer_t, kind, window=10, noise=0.1
):
    """Copy of the function in causal_trace.ipynb"""
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, num_layers):
            layerlist = [
                (tnum, layername(model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            r = trace_with_patch(
                model, inp, layerlist, answer_t, tokens_to_mix=e_range, noise=noise
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def plot_averages(differences, names_and_counts, low_score, modelname, kind, savepdf):
    window = 10
    fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
    h = ax.pcolor(
        differences,
        cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds"}[
            kind
        ],
        vmin=low_score,
    )
    ax.invert_yaxis()
    ax.set_yticks([0.5 + i for i in range(len(differences))])
    ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
    ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
    ax.set_yticklabels([f"{n} ({c})" for n, c in names_and_counts])
    if not kind:
        ax.set_title("Impact of restoring state after corrupted input")
        ax.set_xlabel(f"single restored layer within {modelname}")
    else:
        kindname = "MLP" if kind == "mlp" else "Attn"
        ax.set_title(
            "Avg. impact of restoring {} after corrupted input".format(kindname)
        )
        ax.set_xlabel(f"Center of interval of {window} restored {kindname} layers")
    plt.colorbar(h)
    if savepdf:
        os.makedirs(os.path.dirname(savepdf), exist_ok=True)
        plt.savefig(savepdf, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def get_memorized_ds(dataset_name, eval_df_filename):
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
        start_index = row["start_answer"].item()
        if start_index != 0:
            try:
                example["query_inference"] = (
                    example["query"] + row["prediction"][:start_index]
                )
            except Exception as e:
                print("row", row)
                print("start_index", start_index)
                print("prediction", row["prediction"])
                raise (e)
        else:
            example["query_inference"] = example["query"]
        return example

    def is_trivial_example(ex):
        objs = ex["ground_truth"]
        if not isinstance(objs, list):
            objs = [objs]
        query = ex["query"].lower()
        for possible_ans in objs:
            if possible_ans.lower() in query:
                return True
        return False

    inference_df = pd.read_json(eval_df_filename)
    memorized_df = inference_df[inference_df.exact_match].copy()
    ds = load_dataset(dataset_name)["train"]
    ds = ds.filter(lambda ex: ex["id"] in set(memorized_df["id"].values))

    # Check how many trivial.
    ds_id_to_index = {ex["id"]: i for i, ex in enumerate(ds)}
    memorized_df["query"] = memorized_df.apply(
        lambda row: ds[ds_id_to_index[row["id"]]]["query"], axis=1
    )
    memorized_df["is_trivial"] = memorized_df.apply(
        lambda row: is_trivial_example(row), axis=1
    )
    wandb.log(
        {
            "trivial_memorization": len(memorized_df[memorized_df.is_trivial])
            / len(memorized_df)
        }
    )

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


def plot_hidden_flow(mt, ds, cache_output_dir, pdf_output_dir, kind, noise_level):
    for ex in tqdm(ds, desc="Examples"):
        ex_id = ex["id"]
        filename = os.path.join(cache_output_dir, f"{ex_id}{kind}.npz")
        if not os.path.isfile(filename):
            result = calculate_hidden_flow(
                mt,
                ex["query_inference"],
                ex["sub_label"],
                noise=noise_level,
                kind=kind,
            )
            numpy_result = {
                k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                for k, v in result.items()
            }
            np.savez(filename, **numpy_result)
        else:
            numpy_result = np.load(filename, allow_pickle=True)
        plot_result = dict(numpy_result)
        plot_result["kind"] = kind
        pdfname = os.path.join(
            pdf_output_dir, f'{str(numpy_result["answer"]).strip()}_{ex_id}_{kind}.pdf'
        )
        plot_trace_heatmap(numpy_result, savepdf=pdfname, modelname=args.model_name)

    # Save plot of average.
    total_scores = collections.defaultdict(list)
    files = glob(os.path.join(cache_output_dir, f"*{kind}.npz"))
    for results_file in files:
        numpy_result = np.load(
            os.path.join(cache_output_dir, results_file), allow_pickle=True
        )
        first_subj_token = numpy_result["subject_range"][0]
        last_subj_token = numpy_result["subject_range"][-1] - 1
        total_scores["last_subj_token"].append(numpy_result["scores"][last_subj_token])
        total_scores["first_subj_token"].append(
            numpy_result["scores"][first_subj_token]
        )
        for i in range(0, numpy_result["subject_range"][0]):
            total_scores["before_subj"].append(numpy_result["scores"][i])
        for i in range(first_subj_token + 1, last_subj_token):
            total_scores["mid_subj_tokens"].append(numpy_result["scores"][i])
        for i in range(last_subj_token, len(numpy_result["scores"]) - 1):
            total_scores["after_subj"].append(numpy_result["scores"][i])
        for i in range(len(numpy_result["scores"]) - 1):
            total_scores["after_subj_last"].append(
                numpy_result["scores"][last_subj_token]
            )
        total_scores["low_score"].append(numpy_result["low_score"])
    plot_averages(
        np.array(
            [
                np.mean(total_scores["before_subj"], axis=0),
                np.mean(total_scores["first_subj_token"], axis=0),
                np.mean(total_scores["mid_subj_tokens"], axis=0),
                np.mean(total_scores["last_subj_token"], axis=0),
                np.mean(total_scores["after_subj"], axis=0),
                np.mean(total_scores["after_subj_last"], axis=0),
            ]
        ),
        [
            (k, len(total_scores[k]))
            for k in [
                "before_subj",
                "first_subj_token",
                "mid_subj_tokens",
                "last_subj_token",
                "after_subj",
                "after_subj_last",
            ]
        ],
        np.mean(total_scores["low_score"]),
        args.model_name,
        kind,
        savepdf=os.path.join(pdf_output_dir, f"avg_{kind}.pdf"),
    )
    print(
        "Biggest effect on {} on average in layer {}".format(
            kind, np.argmax(np.mean(total_scores["last_subj_token"], axis=0))
        )
    )
    wandb.summary[f"{kind}_avg_best_layer"] = np.argmax(
        np.mean(total_scores["last_subj_token"], axis=0)
    )


def main(args):
    data_id = "_".join([args.language, args.dataset_name.split("/")[1]])
    cache_output_dir = os.path.join(
        args.output_folder, args.model_name, data_id, "cache_hidden_flow"
    )
    pdf_output_dir = os.path.join(args.output_folder, args.model_name, data_id, "plots")
    wandb.config["cache_output_dir"] = cache_output_dir
    wandb.config["plots_output_dir"] = pdf_output_dir
    os.makedirs(cache_output_dir, exist_ok=True)
    os.makedirs(pdf_output_dir, exist_ok=True)

    mt = load_model_and_tok(args)
    print("Testing prediction...")
    print(
        predict_token(
            mt,
            ["Megan Rapinoe plays the sport of", "The Space Needle is in the city of"],
            return_p=True,
        )
    )

    print("Computing noise level...")
    eval_df_filename = os.path.join(
        args.eval_dir,
        f"{args.language}--{args.dataset_name.split('/')[1]}--{args.model_name}",
        "eval_per_example_records.json",
    )
    ds = get_memorized_ds(args.dataset_name, eval_df_filename)
    print("Computing causal analysis for", len(ds))
    noise_level = 3 * collect_embedding_std(
        mt,
        [ex["sub_label"] for ex in ds],
        subjects_from_ds=data_id,
    )
    print(f"Using noise level {noise_level}")
    for kind in [None, "mlp", "attn"]:
        print("Computing for", kind)
        plot_hidden_flow(mt, ds, cache_output_dir, pdf_output_dir, kind, noise_level)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference")
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
        "--dataset_name",
        required=True,
        type=str,
        help="",
    )
    parser.add_argument(
        "--eval_dir",
        required=True,
        type=str,
        help="",
    )
    parser.add_argument(
        "--language",
        type=str,
        help="",
    )
    args = parser.parse_args()
    if not args.model_name:
        args.model_name = args.model_name_or_path.replace("/", "__")
    wandb.init(
        project="causal_analysis_mpararel",
        name=" ".join([args.model_name, args.language]),
        config=args,
    )
    main(args)
