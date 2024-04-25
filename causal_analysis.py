import argparse
import collections
import os
import string
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
import wandb
from tqdm import tqdm
from dataset.data_utils import get_memorized_dataset, find_token_range
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
from model_utils import load_model_and_tok
from third_party.rome.util import nethook

torch.set_grad_enabled(False)


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
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject, prompt)
    # Add noise and make a forward pass.
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


def plot_averages(
    differences, names_and_counts, low_score, high_score, modelname, kind, savepdf
):
    def _plot_averages(pdf_filename, use_low_score_for_min=True):
        window = 10
        fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
        ticks = np.array(
            [
                differences.min(),
                low_score,
                differences.max(),
            ]
        )
        tick_labels = np.array(
            [
                "{:0.3} {}".format(ticks[i], label)
                for i, label in enumerate(["(Min)", "(Noise)", "(Max)"])
            ]
        )
        args = {"vmin": min(ticks)}
        if use_low_score_for_min:
            args = {"vmin": low_score}
        h = ax.pcolor(
            differences,
            cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds"}[
                kind
            ],
            **args,
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
        cb = plt.colorbar(h)
        cb.ax.set_title("p(%)={:0.3}".format(high_score), y=-0.16, fontsize=10)
        if not use_low_score_for_min:
            cb.set_ticks(ticks[np.argsort(ticks)])
            cb.set_ticklabels(tick_labels[np.argsort(ticks)])
        os.makedirs(os.path.dirname(pdf_filename), exist_ok=True)
        plt.savefig(pdf_filename, bbox_inches="tight")
        plt.close()

    _plot_averages(
        os.path.join(os.path.dirname(savepdf), "ticks_" + os.path.basename(savepdf)),
        use_low_score_for_min=False,
    )
    _plot_averages(savepdf)


def plot_average_trace_heatmap(
    ds, cache_output_dir, pdf_output_dir, kind, model_name, tokenizer
):
    total_scores = collections.defaultdict(list)
    for ex in tqdm(ds, desc="Average Examples"):
        results_file = os.path.join(cache_output_dir, f"{ex['id']}{kind}.npz")
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
            if i == 0 and tokenizer.bos_token is not None:
                total_scores["bos"].append(numpy_result["scores"][i])
            else:
                total_scores["before_subj"].append(numpy_result["scores"][i])
        for i in range(first_subj_token + 1, last_subj_token):
            total_scores["mid_subj_tokens"].append(numpy_result["scores"][i])
        for i in range(first_subj_token, last_subj_token + 1):
            total_scores["all_subj_tokens"].append(numpy_result["scores"][i])
        if last_subj_token < len(numpy_result["scores"]) - 1:
            for i in range(last_subj_token, len(numpy_result["scores"])):
                total_scores["after_subj"].append(numpy_result["scores"][i])
            total_scores["after_subj_first"].append(numpy_result["scores"][0])
            total_scores["after_subj_last"].append(numpy_result["scores"][-1])
        total_scores["last_token"].append(numpy_result["scores"][-1])
        total_scores["low_score"].append(numpy_result["low_score"])
        total_scores["high_score"].append(numpy_result["high_score"])
    agg_tokens_keys = [
        "bos",
        "before_subj",
        "first_subj_token",
        "mid_subj_tokens",
        "last_subj_token",
        "all_subj_tokens",
        "after_subj",
        "after_subj_last",
        "last_token",
    ]
    differences = np.array(
        [
            np.mean(total_scores[k], axis=0)
            for k in agg_tokens_keys
            if len(total_scores[k]) > 0
        ]
    )
    plot_averages(
        differences,
        [
            (k, len(total_scores[k]))
            for k in agg_tokens_keys
            if len(total_scores[k]) > 0
        ],
        np.mean(total_scores["low_score"]),
        np.mean(total_scores["high_score"]),
        model_name,
        kind,
        savepdf=os.path.join(pdf_output_dir, f"avg_{kind}.pdf"),
    )
    print(
        "Biggest effect on {} on average in layer {}".format(
            kind, np.argmax(np.mean(total_scores["last_subj_token"], axis=0))
        )
    )
    if wandb.run is not None:
        wandb.summary[f"{kind}_avg_best_layer"] = np.argmax(
            np.mean(total_scores["last_subj_token"], axis=0)
        )
        p_noise = np.mean(total_scores["low_score"])
        p_min = differences.min()
        wandb.summary[f"significant_{kind}"] = (
            p_noise + (p_noise - p_min) < differences.max()
        )


def plot_hidden_flow(
    mt,
    ds,
    cache_output_dir,
    pdf_output_dir,
    kind,
    noise_level,
    recompute_query_inference=False,
):
    for ex in tqdm(ds, desc="Examples"):
        ex_id = ex["id"]
        filename = os.path.join(cache_output_dir, f"{ex_id}{kind}.npz")
        if not os.path.isfile(filename) or (
            recompute_query_inference and ex["query_inference"] != ex["query"]
        ):
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
        plot_trace_heatmap(numpy_result, savepdf=pdfname, modelname=mt.model_name)

    # Save plot of average.
    plot_average_trace_heatmap(
        ds, cache_output_dir, pdf_output_dir, kind, mt.model_name
    )


def main(args):
    data_id = "_".join(
        [
            args.language,
            (
                args.dataset_name
                if "/" not in args.dataset_name
                else args.dataset_name.split("/")[1]
            ),
        ]
    )
    if args.only_subset:
        data_id = data_id + "_subset"
    cache_dir = os.path.join(args.output_folder, args.model_name, data_id)
    if args.override_noise_level is not None:
        cache_dir = os.path.join(
            args.output_folder,
            args.model_name,
            data_id + f"_noise={args.override_noise_level}",
        )
    cache_hidden_flow = os.path.join(cache_dir, "cache_hidden_flow")
    pdf_output_dir = os.path.join(cache_dir, "plots")
    wandb.config["cache_output_dir"] = cache_dir
    wandb.config["plots_output_dir"] = pdf_output_dir
    os.makedirs(cache_hidden_flow, exist_ok=True)
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

    ds = get_memorized_dataset(
        args.dataset_name,
        args.language,
        args.eval_dir,
        args.model_name,
        args.only_subset,
    )
    print("Computing causal analysis for", len(ds))

    print("Computing noise level...")
    if args.override_noise_level is not None:
        noise_level = args.override_noise_level
    else:
        noise_level = 3 * collect_embedding_std(
            mt,
            [ex["sub_label"] for ex in ds],
            subjects_from_ds=data_id,
        )
    wandb.run.summary["noise_level"] = noise_level
    print(f"Using noise level {noise_level}")
    for kind in [None, "mlp", "attn"]:
        print("Computing for", kind)
        if args.only_plot_average:
            plot_average_trace_heatmap(
                ds, cache_hidden_flow, pdf_output_dir, kind, mt.model_name
            )
            continue
        plot_hidden_flow(
            mt,
            ds,
            cache_hidden_flow,
            pdf_output_dir,
            kind,
            noise_level,
            recompute_query_inference=args.recompute_query_inference,
        )


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
        type=str,
        help="",
    )
    parser.add_argument(
        "--language",
        type=str,
        help="",
    )
    parser.add_argument("--only_subset", action="store_true")
    parser.add_argument("--override_noise_level", type=float, help="")
    parser.add_argument("--only_plot_average", action="store_true")
    parser.add_argument("--recompute_query_inference", action="store_true")
    args = parser.parse_args()
    if not args.model_name:
        args.model_name = args.model_name_or_path.replace("/", "__")
    wandb.init(
        project="causal_analysis_mpararel",
        name=" ".join(
            [
                args.model_name,
                args.language,
                (
                    f"noise={args.override_noise_level}"
                    if args.override_noise_level is not None
                    else ""
                ),
            ]
        ),
        config=args,
    )
    main(args)
