import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import wandb
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    XGLMForCausalLM,
)

from dataset.data_utils import get_memorized_dataset, find_token_range
from third_party.rome.experiments.causal_trace import layername
from third_party.rome.util.nethook import get_module


def get_hidden_state_from_output(model, output, tok_index, output_type):
    if isinstance(model, XGLMForCausalLM) or isinstance(model, GPT2LMHeadModel):
        if output_type.startswith("attn") or output_type.startswith("out"):
            # This output is a tuple.
            output = output[0]
        return output[:, tok_index].detach()
    raise LookupError("Add {} to lookup.".format(type(model)))


def set_act_get_hooks(
    model, total_layers, tok_index, hook_attn=False, hook_mlp=False, hook_out=True
):
    for attr in ["activations_"]:
        if not hasattr(model, attr):
            setattr(model, attr, {})

    def get_activation(name):
        def hook(module, input, output):
            model.activations_[name] = get_hidden_state_from_output(
                model, output, tok_index, name
            )

        return hook

    hooks = []
    for i in range(total_layers):
        if hook_attn:
            hooks.append(
                get_module(model, layername(model, i, "attn")).register_forward_hook(
                    get_activation(f"attn_{i}")
                )
            )
        if hook_mlp:
            hooks.append(
                get_module(model, layername(model, i, "mlp")).register_forward_hook(
                    get_activation(f"mlp_{i}")
                )
            )
        if hook_out:
            hooks.append(
                get_module(model, layername(model, i, None)).register_forward_hook(
                    get_activation(f"out_{i}")
                )
            )

    return hooks


def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()


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
        wandb.run.name += " subset"
        data_id = data_id + "_subset"
    if args.filter_trivial:
        wandb.run.name += " filter_trivial"
        data_id += "_filter_trivial"
    if args.resample_trivial:
        wandb.run.name += " resample_trivial"
        data_id += "_resample_trivial"
    if args.keep_only_trivial:
        wandb.run.name += " only_trivial"
        data_id += "_only_trivial"
    if args.last_subject_token:
        wandb.run.name += " last_subject_token"
        data_id += "_last_subject_token"
    if args.store_topk:
        wandb.run.name = f"(top{args.store_topk}) {wandb.run.name}"
        args.output_folder = os.path.normpath(args.output_folder)
        args.output_folder = os.path.join(
            os.path.dirname(args.output_folder),
            os.path.basename(args.output_folder) + f"_top{args.store_topk}",
        )
    args.output_folder = os.path.join(args.output_folder, args.model_name, data_id)
    wandb.config["final_output_dir"] = args.output_folder
    os.makedirs(args.output_folder, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

    total_layers = len(get_module(model, layername(model, kind="layers")))
    lm_head = get_module(model, layername(model, kind="lm_head")).weight

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

    records = []
    for ex in tqdm(ds, desc="Examples"):
        text_input = ex["query_inference"]
        inp = tokenizer(text_input, return_tensors="pt").to(device)
        last_token_index = inp["input_ids"].shape[1] - 1
        if args.last_subject_token:
            subj_range = find_token_range(tokenizer, inp["input_ids"], ex["sub_label"])
            last_token_index = subj_range[1] - 1
        hooks = set_act_get_hooks(
            model,
            total_layers,
            last_token_index,
            hook_mlp="mlp_" in args.hook_modules,
            hook_attn="attn_" in args.hook_modules,
            hook_out="out_" in args.hook_modules,
        )
        with torch.no_grad():
            outputs = model(**inp)
        remove_hooks(hooks)
        token_pred = torch.argmax(outputs.logits[0, -1, :]).item()

        for layer in range(total_layers):
            for k in args.hook_modules:
                out = model.activations_[f"{k}{layer}"][0]
                proj = (lm_head @ out).detach().cpu().numpy()
                ind = np.argsort(-proj, axis=-1)  # Descending order.
                pred_tok_rank = np.where(ind == token_pred)[0][0]
                extra_items = {}
                if args.store_topk:
                    extra_items = {
                        "topk_token_ids": ind[: args.store_topk],
                        "topk_tokens": tokenizer.convert_ids_to_tokens(
                            ind[: args.store_topk]
                        ),
                    }
                records.append(
                    {
                        "id": ex["id"],
                        "language": ex["id"].split("_")[0],
                        "relation": ex["id"].split("_")[1],
                        "layer": layer,
                        "prompt": text_input,
                        "subject": ex["sub_label"],
                        "last_token_index": last_token_index,
                        "final_pred": token_pred,
                        "proj_vec": k,
                        "pred_tok_rank": pred_tok_rank,
                        "pred_tok_score": proj[ind[pred_tok_rank]],
                        "pred_in_top_1": pred_tok_rank == 0,
                        **extra_items,
                    }
                )
    df = pd.DataFrame(records)
    filename = "extraction_events"
    df.to_csv(os.path.join(args.output_folder, f"{filename}.csv"), index=False)
    with open(os.path.join(args.output_folder, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)
    df[["proj_vec", "language", "relation", "layer", "pred_in_top_1"]].groupby(
        by=["proj_vec", "language", "relation", "layer"], as_index=False
    ).mean().to_csv(
        os.path.join(args.output_folder, f"{filename}_avg_relation.csv"),
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--model_name", type=str)
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
    parser.add_argument("--hook_modules", nargs="+", default=["attn_", "mlp_", "out_"])
    parser.add_argument("--store_topk", type=int)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--last_subject_token", action="store_true")
    args = parser.parse_args()

    if not args.model_name:
        args.model_name = args.model_name_or_path.replace("/", "__")

    run_name = "{} {}".format(args.model_name_or_path, args.language)
    if "WANDB_NAME" in os.environ:
        run_name = os.getenv("WANDB_NAME")
    wandb.init(project="xfact_extraction_rate", name=run_name, config=args)

    main(args)
