import argparse
import wandb
from tqdm import tqdm
import os
from dataset.data_utils import get_memorized_dataset, find_token_range
from inference.run_inference import prepare_prompt
from third_party.rome.experiments.causal_trace import (
    decode_tokens,
    layername,
    predict_from_input,
    predict_token,
)
import torch
from model_utils import load_model_and_tok
from patching_utils import (
    trace_important_states,
    trace_important_window,
    trace_with_patch,
)


def get_token_indices(token_to_patch, examples, input_ids, input_prompts, tokenizer):
    if token_to_patch == "last":
        token_idx_to_patch_from = -1
        token_idx_to_patch = -1
    elif token_to_patch == "last_subject_token":
        subj_ranges = []
        for ex, inp, prompt in zip(examples, input_ids, input_prompts):
            subj_ranges.append(find_token_range(tokenizer, inp, ex["subject"], prompt))
        token_idx_to_patch_from = subj_ranges[0][-1]
        token_idx_to_patch = subj_ranges[1][-1]
    return token_idx_to_patch_from, token_idx_to_patch


def patch_ex1_into_ex2(
    mt, ex1, ex2, layers_to_patch, stack_to_patch, kind, token_to_patch="last"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_prompts = []
    for ex in [ex1, ex2]:
        prompt = ex["query_inference"]
        input_prompts.append(
            prepare_prompt(prompt, args.model_name_or_path, "", "t5" in mt.model_name)
        )
    # TODO: add padding
    inp = mt.tokenizer(input_prompts, return_tensors="pt").to(device)
    token_idx_to_patch_from, token_idx_to_patch = get_token_indices(
        token_to_patch, [ex1, ex2], inp["input_ids"], input_prompts, mt.tokenizer
    )
    # TODO: add the object in the other languages to track as well?
    with torch.no_grad():
        preds_tokens, preds_probs = [d[0] for d in predict_from_input(mt.model, inp)]
    answers = decode_tokens(mt.tokenizer, preds_tokens)
    layers_results = []
    for layer_i in range(0, layers_to_patch):
        # TODO: add option in the function to have a tuple of tokens indices.
        r = trace_with_patch(
            mt.model,
            inp,
            [
                (
                    (token_idx_to_patch_from, token_idx_to_patch),
                    layername(mt.model, stack=stack_to_patch, num=layer_i, kind=kind),
                )
            ],
            preds_tokens,
            tokens_to_mix=None,
            noise=None,
        )
        layers_results.append(r)
    layers_results = torch.stack(layers_results).detach().cpu()
    return dict(
        scores=layers_results,
        high_score=preds_probs,
        input_ids=inp["input_ids"].detach().cpu().numpy(),
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"]),
        answer=answers,
        window=layers_to_patch,
        kind=kind or "",
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
    if args.patch_k_layers != 10:
        data_id += f"_window={args.patch_k_layers}"
    if args.override_noise_level is not None:
        data_id += +f"_noise={args.override_noise_level}"
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

    ds = get_memorized_dataset(
        args.dataset_name,
        args.language,
        args.eval_dir,
        args.model_name,
        args.only_subset,
        mt.tokenizer,
        args.filter_trivial,
        args.resample_trivial,
        args.keep_only_trivial,
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
    parser.add_argument("--filter_trivial", action="store_true")
    parser.add_argument("--keep_only_trivial", action="store_true")
    parser.add_argument("--resample_trivial", action="store_true")
    parser.add_argument("--use_vmin_vmax_from_folder", type=str, default=None)
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
