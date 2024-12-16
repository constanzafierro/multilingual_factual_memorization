import argparse
import collections
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from tqdm import tqdm
from transformers import T5TokenizerFast

from dataset.data_utils import find_token_range, get_memorized_dataset, get_dataset_name
from model_utils import load_model_and_tok
from patching_utils import (
    trace_important_states,
    trace_important_window,
    trace_with_patch,
)
from third_party.rome.experiments.causal_trace import (
    collect_embedding_std,
    decode_tokens,
    plot_trace_heatmap,
    predict_from_input,
    predict_token,
)

torch.set_grad_enabled(False)


def calculate_hidden_flow(
    mt,
    prompt,
    input_ids,
    subject,
    decoder_input_ids=None,
    noise=0.1,
    window=10,
    kind=None,
    samples=10,
    expected_ans=None,
    use_logits=False,
):
    """
    Copy of the function in causal_trace.ipynb
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inp = {
        "input_ids": torch.tensor([input_ids for _ in range(samples + 1)]).to(device),
    }
    inp["attention_mask"] = torch.ones_like(inp["input_ids"])
    if decoder_input_ids is not None:
        decoder_input_ids = torch.tensor(
            [decoder_input_ids for _ in range(samples + 1)]
        ).to(device)
        inp = {**inp, "decoder_input_ids": decoder_input_ids}
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(mt.model, inp)]
    [answer] = decode_tokens(mt.tokenizer, [answer_t])
    if not expected_ans.startswith(answer):
        if mt.tokenizer.unk_token_id in input_ids:
            return None
        else:
            raise Exception(
                "For the prompt='{}', expected to get the beggining of '{}' but"
                "instead got='{}'".format(prompt, expected_ans, answer)
            )
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject, prompt)
    # Add noise and make a forward pass.
    low_score, low_score_logit = trace_with_patch(
        mt.model, inp, [], answer_t, e_range, noise=noise, use_logits=use_logits
    )
    low_score, low_score_logit = low_score.item(), low_score_logit.item()
    if not kind:
        prob_diffs, logit_diffs = trace_important_states(
            mt.model,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            use_logits=use_logits,
        )
    else:
        prob_diffs, logit_diffs = trace_important_window(
            mt.model,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            window=window,
            kind=kind,
            low_score=low_score,
            use_logits=use_logits,
        )
    prob_diffs = prob_diffs.detach().cpu()
    logit_diffs = logit_diffs.detach().cpu()
    input_ids = inp["input_ids"][0]
    if decoder_input_ids is not None:
        input_ids = np.concatenate(
            (
                inp["input_ids"][0].detach().cpu().numpy(),
                inp["decoder_input_ids"][0].detach().cpu().numpy(),
            )
        )
    return dict(
        scores=prob_diffs,
        logit_diffs=logit_diffs,
        low_score=low_score,
        low_score_logit=low_score_logit,
        high_score=base_score,
        input_ids=input_ids,
        input_tokens=decode_tokens(mt.tokenizer, input_ids),
        subject_range=e_range,
        answer=answer,
        window=window,
        kind=kind or "",
    )


def plot_averages(
    scores,
    names_and_counts,
    low_score,
    high_score,
    modelname,
    kind,
    savepdf,
    vmin_vmax=None,
):
    def _plot_averages(pdf_filename, use_min_for_vmin=False, vmin_vmax=None):
        window = 10
        fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
        ticks = np.array(
            [
                scores.min(),
                low_score,
                scores.max(),
            ]
        )
        tick_labels = np.array(
            [
                "{:0.3} {}".format(ticks[i], label)
                for i, label in enumerate(["(Min)", "(Noise)", "(Max)"])
            ]
        )
        args = {"vmin": low_score}
        if vmin_vmax is not None:
            args = {"vmin": vmin_vmax[0], "vmax": vmin_vmax[1]}
        if use_min_for_vmin:
            args = {"vmin": min(ticks)}
        h = ax.pcolor(
            scores,
            cmap={
                None: "Purples",
                "None": "Purples",
                "mlp": "Greens",
                "attn": "Reds",
                "cross_attn": "Blues",
            }[kind],
            **args,
        )
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(scores))])
        ax.set_xticks([0.5 + i for i in range(0, scores.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, scores.shape[1] - 6, 5)))
        ax.set_yticklabels([f"{n} ({c})" for n, c in names_and_counts])
        if not kind:
            ax.set_title("Impact of restoring state after corrupted input")
            ax.set_xlabel(f"single restored layer within {modelname}")
        else:
            ax.set_title(
                "Avg. impact of restoring {} after corrupted input".format(kind)
            )
            ax.set_xlabel(f"Center of interval of {window} restored {kind} layers")
        cb = plt.colorbar(h)
        cb.ax.set_title("p(%)={:0.3}".format(high_score), y=-0.16, fontsize=10)
        if use_min_for_vmin:
            cb.set_ticks(ticks[np.argsort(ticks)])
            cb.set_ticklabels(tick_labels[np.argsort(ticks)])
        os.makedirs(os.path.dirname(pdf_filename), exist_ok=True)
        plt.savefig(pdf_filename, bbox_inches="tight")
        plt.close()

    _plot_averages(
        os.path.join(os.path.dirname(savepdf), "ticks_" + os.path.basename(savepdf)),
        use_min_for_vmin=True,
    )
    if vmin_vmax is not None:
        _plot_averages(
            os.path.join(
                os.path.dirname(savepdf), "vmin_vmax_" + os.path.basename(savepdf)
            ),
            vmin_vmax=vmin_vmax,
        )
        _plot_averages(savepdf, vmin_vmax=None)
    else:
        _plot_averages(savepdf, vmin_vmax=vmin_vmax)


def agg_causal_analysis_results(tokenizer, ds, cache_output_dir, kind, missing_ids):
    tok_special_ids = set(tokenizer.all_special_ids)
    has_bos = tokenizer("some long text here")["input_ids"][0] in tok_special_ids
    has_eos = tokenizer("some long text here")["input_ids"][-1] in tok_special_ids
    mask_token = (
        None
        if not isinstance(tokenizer, T5TokenizerFast)
        else tokenizer("<extra_id_0>", add_special_tokens=False)["input_ids"][0]
    )
    after_subj_last = "after_subj_last"
    if has_eos:
        after_subj_last = "eos"
    all_scores = []
    for ex in tqdm(ds, desc="Average Examples"):
        results_file = os.path.join(cache_output_dir, f"{ex['id']}{kind}.npz")
        if f"{ex['id']}{kind}.npz" in missing_ids:
            continue
        numpy_result = np.load(
            os.path.join(cache_output_dir, results_file), allow_pickle=True
        )
        decoder_input_ids = ex["decoder_input_ids"]
        input_ids = numpy_result["input_ids"]
        encoder_scores = numpy_result["scores"]
        if decoder_input_ids:
            encoder_scores = numpy_result["scores"][: -len(decoder_input_ids)]
            decoder_scores = numpy_result["scores"][len(encoder_scores) :]
            decoder_input_ids = input_ids[len(encoder_scores) :]
            input_ids = input_ids[: len(encoder_scores)]
        assert input_ids_match(ex, numpy_result), f"{ex['id']}_{kind}"

        ex_scores = collections.defaultdict(list)
        first_subj_token = numpy_result["subject_range"][0]
        last_subj_token = numpy_result["subject_range"][-1] - 1
        # Subject tokens scores.
        ex_scores["last_subj_token"].append(encoder_scores[last_subj_token])
        if first_subj_token != last_subj_token:
            ex_scores["first_subj_token"].append(encoder_scores[first_subj_token])
        for i in range(first_subj_token + 1, last_subj_token):
            ex_scores["mid_subj_tokens"].append(encoder_scores[i])
        for i in range(first_subj_token, last_subj_token + 1):
            ex_scores["all_subj_tokens"].append(encoder_scores[i])
        # Before subject.
        for i in range(0, numpy_result["subject_range"][0]):
            if i == 0 and has_bos:
                ex_scores["bos"].append(encoder_scores[i])
            elif input_ids[i] == mask_token:
                ex_scores["mask_token"].append(encoder_scores[i])
            else:
                ex_scores["before_subj"].append(encoder_scores[i])
        # After subject.
        if last_subj_token < len(encoder_scores) - 1:
            for i in range(last_subj_token, len(encoder_scores)):
                if input_ids[i] == mask_token:
                    ex_scores["mask_token"].append(encoder_scores[i])
                else:
                    ex_scores["after_subj"].append(encoder_scores[i])
            ex_scores["after_subj_first"].append(encoder_scores[0])
            ex_scores[after_subj_last].append(encoder_scores[-1])
        # Decoder tokens.
        if decoder_input_ids is not None:
            for i in range(0, len(decoder_scores)):
                if decoder_input_ids[i] in tok_special_ids:
                    ex_scores["dec_bos"].append(decoder_scores[i])
                elif decoder_input_ids[i] == mask_token:
                    ex_scores["dec_mask_token"].append(decoder_scores[i])
                else:
                    ex_scores["dec"].append(decoder_scores[i])
        ex_scores["last_token"].append(numpy_result["scores"][-1])
        ex_scores["low_score"].append(numpy_result["low_score"])
        ex_scores["high_score"].append(numpy_result["high_score"])
        all_scores.append(ex_scores)
    return all_scores


def compute_averages(all_scores):
    agg_tokens_keys = [
        "bos",
        "mask_token",
        "before_subj",
        "first_subj_token",
        "mid_subj_tokens",
        "last_subj_token",
        "all_subj_tokens",
        "after_subj_first",
        "after_subj",
        "after_subj_last",
        "eos",
        "dec_bos",
        "dec_mask_token",
        "dec",
        "last_token",
    ]
    scores = []
    scores_macro = []
    differences = []
    differences_macro = []
    counts = []
    final_keys = []
    for k in agg_tokens_keys:
        k_scores = [ex_scores[k] for ex_scores in all_scores if k in ex_scores]
        if not k_scores:
            continue
        final_keys.append(k)
        counts.append(sum([len(l) for l in k_scores]))
        k_diffs = [
            np.array(ex_scores[k]) - ex_scores["low_score"][0]
            for ex_scores in all_scores
            if k in ex_scores
        ]
        scores.append(np.mean([i for l in k_scores for i in l], axis=0))
        differences.append(np.mean([i for l in k_diffs for i in l], axis=0))
        scores_macro.append(np.mean([np.mean(s, axis=0) for s in k_scores], axis=0))
        differences_macro.append(np.mean([np.mean(s, axis=0) for s in k_diffs], axis=0))
    scores = np.array(scores)
    scores_macro = np.array(scores_macro)
    differences = np.array(differences)
    differences_macro = np.array(differences_macro)
    return {
        "scores": np.array(scores),
        "scores_macro": np.array(scores_macro),
        "differences": np.array(differences),
        "differences_macro": np.array(differences_macro),
        "agg_tokens_keys": final_keys,
        "counts": counts,
        "low_score": np.mean([ex["low_score"] for ex in all_scores]),
        "high_score": np.mean([ex["high_score"] for ex in all_scores]),
    }


def plot_average_trace_heatmap(
    ds,
    cache_output_dir,
    pdf_output_dir,
    kind,
    model_name,
    tokenizer,
    use_vmin_vmax_from_folder=None,
    missing_ids=None,
):
    all_scores = agg_causal_analysis_results(
        tokenizer, ds, cache_output_dir, kind, missing_ids
    )
    averaged_scores = compute_averages(all_scores)
    names_and_counts = list(
        zip(averaged_scores["agg_tokens_keys"], averaged_scores["counts"])
    )

    np.savez(
        os.path.join(pdf_output_dir, f"avg_data_{kind}.npz"),
        **{
            "names_and_counts": names_and_counts,
            "model_name": model_name,
            "kind": kind,
            **averaged_scores,
        },
    )

    vmin_max = None
    if use_vmin_vmax_from_folder is not None:
        numpy_result = np.load(
            os.path.join(use_vmin_vmax_from_folder, f"avg_data_{kind}.npz"),
            allow_pickle=True,
        )
        vmin = numpy_result["low_score"]
        vmax = numpy_result["scores"].max()
        vmin_max = [vmin, vmax]
    plot_averages(
        averaged_scores["scores"],
        names_and_counts,
        averaged_scores["low_score"],
        averaged_scores["high_score"],
        model_name,
        kind,
        savepdf=os.path.join(pdf_output_dir, f"avg_{kind}.pdf"),
        vmin_vmax=vmin_max,
    )
    biggest_effect = np.argmax(
        np.mean([ex["last_subj_token"] for ex in all_scores], axis=0)
    )
    print(
        "Biggest effect on {} on average in layer {}".format(
            kind,
            biggest_effect,
        )
    )
    if wandb.run is not None:
        wandb.summary[f"{kind}_avg_best_layer"] = biggest_effect
        p_noise = (np.mean([ex["low_score"] for ex in all_scores]),)
        p_min = averaged_scores["scores"].min()
        wandb.summary[f"significant_{kind}"] = (
            p_noise + (p_noise - p_min) < averaged_scores["scores"].max()
        )


def input_ids_match(ex, numpy_result):
    if ex["decoder_input_ids"] is not None:
        decoder_input_ids = numpy_result["input_ids"][-len(ex["decoder_input_ids"]) :]
        input_ids = numpy_result["input_ids"][: -len(ex["decoder_input_ids"])]
        return (
            len(ex["decoder_input_ids"]) == len(decoder_input_ids)
            and np.all(ex["decoder_input_ids"] == decoder_input_ids)
            and len(ex["input_ids"]) == len(input_ids)
            and np.all(ex["input_ids"] == input_ids)
        )
    return len(numpy_result["input_ids"]) == len(ex["input_ids"]) and np.all(
        numpy_result["input_ids"] == ex["input_ids"]
    )


def plot_hidden_flow(
    mt,
    ds,
    cache_output_dir,
    pdf_output_dir,
    kind,
    noise_level,
    patch_k_layers,
    use_logits,
    override=False,
):
    unk_in_query_inference = set()
    for ex in tqdm(ds, desc="Examples"):
        ex_id = ex["id"]
        filename = os.path.join(cache_output_dir, f"{ex_id}{kind}.npz")
        if os.path.isfile(filename):
            numpy_result = np.load(filename, allow_pickle=True)
        if not os.path.isfile(filename) or (
            override and not input_ids_match(ex, numpy_result)
        ):
            result = calculate_hidden_flow(
                mt=mt,
                prompt=ex["query_inference"],
                input_ids=ex["input_ids"],
                subject=ex["sub_label"],
                decoder_input_ids=ex["decoder_input_ids"],
                noise=noise_level,
                kind=kind,
                window=patch_k_layers,
                expected_ans=ex["prediction"],
                use_logits=use_logits,
            )
            if not result:
                unk_in_query_inference.add(f"{ex_id}{kind}.npz")
                continue
            numpy_result = {
                k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                for k, v in result.items()
            }
            np.savez(filename, **numpy_result)
        plot_result = dict(numpy_result)
        plot_result["kind"] = kind
        pdfname = os.path.join(
            pdf_output_dir, f'{str(numpy_result["answer"]).strip()}_{ex_id}_{kind}.pdf'
        )
        plot_trace_heatmap(numpy_result, savepdf=pdfname, modelname=mt.model_name)
    return unk_in_query_inference


def main(args):
    dataset_name = get_dataset_name(args.model_name, args.language)
    data_id = "_".join(
        [
            args.language,
            (dataset_name if "/" not in dataset_name else dataset_name.split("/")[1]),
        ]
    )
    if args.only_subset:
        data_id = data_id + "_subset"
    if args.patch_k_layers != 10:
        data_id += f"_window={args.patch_k_layers}"
    if args.override_noise_level is not None:
        data_id += f"_noise={args.override_noise_level}"
    if args.use_logits:
        data_id += "_logits"
    cache_dir = os.path.join(args.output_folder, args.model_name, data_id)

    cache_hidden_flow = os.path.join(cache_dir, "cache_hidden_flow")
    plots_folder = "plots"
    if args.filter_trivial:
        wandb.run.name += " filter_trivial"
        plots_folder = "plots_filter_trivial"
    if args.resample_trivial:
        wandb.run.name += " resample_trivial"
        plots_folder = "plots_resample_trivial"
    if args.keep_only_trivial:
        wandb.run.name += " only_trivial"
        plots_folder = "plots_only_trivial"
    pdf_output_dir = os.path.join(cache_dir, plots_folder)
    if args.use_vmin_vmax_from_folder is not None:
        args.use_vmin_vmax_from_folder = os.path.join(
            cache_dir, args.use_vmin_vmax_from_folder
        )
    wandb.config["cache_output_dir"] = cache_dir
    wandb.config["plots_output_dir"] = pdf_output_dir
    os.makedirs(cache_hidden_flow, exist_ok=True)
    os.makedirs(pdf_output_dir, exist_ok=True)

    mt = load_model_and_tok(args.model_name_or_path, args.model_name)
    if not args.only_plot_average:
        print("Testing prediction...")
        print(
            predict_token(
                mt,
                [
                    "Megan Rapinoe plays the sport of",
                    "The Space Needle is in the city of",
                ],
                decoder_prompt=(
                    ["<pad> <extra_id_0>"] * 2 if "t5" in args.model_name else None
                ),
                return_p=True,
            )
        )
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
    print("Computing causal analysis for", len(ds))
    wandb.run.summary["n_examples_causal_analysis"] = len(ds)

    print("Computing noise level...")
    if args.override_noise_level is not None:
        noise_level = args.override_noise_level
    else:
        noise_level = 3 * collect_embedding_std(
            mt,
            [ex["sub_label"] for ex in ds],
            seq2seq="t5" in args.model_name,
            subjects_from_ds=data_id,
        )
    wandb.run.summary["noise_level"] = noise_level
    print(f"Using noise level {noise_level}")
    modules = (
        ["cross_attn", None, "mlp", "attn"]
        if hasattr(mt.model, "decoder")
        else [None, "mlp", "attn"]
    )
    failed_ids = set()
    for kind in modules:
        print("Computing for", kind)
        if not args.only_plot_average:
            failed_ids.update(
                plot_hidden_flow(
                    mt,
                    ds,
                    cache_hidden_flow,
                    pdf_output_dir,
                    kind,
                    noise_level,
                    args.patch_k_layers,
                    use_logits=args.use_logits,
                    override=args.override,
                )
            )
        plot_average_trace_heatmap(
            ds,
            cache_hidden_flow,
            pdf_output_dir,
            kind,
            mt.model_name,
            mt.tokenizer,
            args.use_vmin_vmax_from_folder,
            failed_ids,
        )
    wandb.log({"unk_in_query_inference": len(failed_ids)})


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
        "--eval_dir",
        type=str,
        help="",
    )
    parser.add_argument(
        "--language",
        type=str,
        help="",
    )
    parser.add_argument("--patch_k_layers", type=int, default=10)
    parser.add_argument("--only_subset", action="store_true")
    parser.add_argument("--override_noise_level", type=float, help="")
    parser.add_argument("--only_plot_average", action="store_true")
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--filter_trivial", action="store_true")
    parser.add_argument("--keep_only_trivial", action="store_true")
    parser.add_argument("--resample_trivial", action="store_true")
    parser.add_argument("--use_vmin_vmax_from_folder", type=str, default=None)
    parser.add_argument("--use_logits", action="store_true")
    args = parser.parse_args()
    if not args.model_name:
        args.model_name = args.model_name_or_path.replace("/", "__")
    wandb.init(
        project="causal_analysis_mpararel",
        name=" ".join(
            [
                args.model_name,
                args.language,
                ("(logits)" if args.use_logits else ""),
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
