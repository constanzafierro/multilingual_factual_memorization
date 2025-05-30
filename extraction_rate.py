import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd
import torch
import wandb
from tqdm import tqdm
from transformers import (
    GPT2LMHeadModel,
    MT5ForConditionalGeneration,
    XGLMForCausalLM,
    LlamaForCausalLM,
)

from dataset.data_utils import find_token_range, get_dataset_name, get_memorized_dataset
from model_utils import load_model_and_tok
from third_party.rome.experiments.causal_trace import layername
from third_party.rome.util.nethook import get_module


def get_hidden_state_from_output(model, output, tok_index, output_type):
    if type(model) in {
        XGLMForCausalLM,
        GPT2LMHeadModel,
        MT5ForConditionalGeneration,
        LlamaForCausalLM,
    }:
        if not output_type.startswith("mlp"):
            # The first position of the output tuple contains the hidden
            output = output[0]
        return output[:, tok_index].detach()
    raise LookupError(
        "Add condition for {} to obtain the hidden state from the output of the"
        "module.".format(type(model))
    )


def set_act_get_hooks(model, total_layers, tok_index, hook_modules):
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
        for module in hook_modules:
            kind = module
            if module == "out":
                kind = None
            hooks.append(
                get_module(
                    model,
                    layername(model=model, num=i, kind=kind, stack="decoder"),
                ).register_forward_hook(get_activation(f"{module}_{i}"))
            )

    return hooks


def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()


def main(args):
    data_id = "_".join(
        [
            args.language,
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
    if args.apply_layer_norm:
        wandb.run.name += " apply_layer_norm"
        data_id += "_apply_layer_norm"
    if args.store_topk:
        wandb.run.name = f"(top{args.store_topk}) {wandb.run.name}"
        args.output_folder = os.path.normpath(args.output_folder)
        args.output_folder = os.path.join(
            os.path.dirname(args.output_folder),
            os.path.basename(args.output_folder) + f"_top{args.store_topk}",
        )
        os.makedirs(os.path.join(args.output_folder, "tmp"))
    args.output_folder = os.path.join(args.output_folder, args.model_name, data_id)
    wandb.config["final_output_dir"] = args.output_folder
    os.makedirs(args.output_folder, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mt = load_model_and_tok(args.model_name_or_path, args.model_name)
    model, tokenizer = mt.model, mt.tokenizer

    total_layers = len(get_module(model, layername(model, kind="layers")))
    lm_head = get_module(model, layername(model, kind="lm_head")).weight
    layer_norm = (
        None
        if not args.apply_layer_norm
        else get_module(model, layername(model, kind="lm_norm"))
    )

    dataset_name = get_dataset_name(args.model_name, args.language)
    ds = get_memorized_dataset(
        dataset_name,
        args.language,
        args.eval_dir,
        args.model_name,
        args.only_subset,
        mt.tokenizer,
        filter_trivial=args.filter_trivial,
        resample_trivial=args.resample_trivial,
        keep_only_trivial=args.keep_only_trivial,
    )

    records = []
    tmp_stored_records = []
    for ex in tqdm(ds, desc="Examples"):
        text_input = ex["query_inference"]
        inp = {"input_ids": torch.tensor([ex["input_ids"]]).to(device)}
        inp["attention_mask"] = torch.ones_like(inp["input_ids"])
        last_token_index = inp["input_ids"].shape[1] - 1
        if ex["decoder_input_ids"] is not None:
            decoder_input_ids = torch.tensor([ex["decoder_input_ids"]]).to(device)
            inp = {**inp, "decoder_input_ids": decoder_input_ids}
            last_token_index = inp["decoder_input_ids"].shape[1] - 1
        if args.last_subject_token:
            subj_range = find_token_range(
                tokenizer, inp["input_ids"][0], ex["sub_label"], text_input
            )
            last_token_index = subj_range[1] - 1
        if not args.hook_modules:
            args.hook_modules = (
                ["cross_attn", "out", "mlp", "attn"]
                if hasattr(model, "decoder")
                else ["out", "mlp", "attn"]
            )
        hooks = set_act_get_hooks(
            model,
            total_layers,
            last_token_index,
            hook_modules=args.hook_modules,
        )
        with torch.no_grad():
            outputs = model(**inp)
        remove_hooks(hooks)
        token_pred = torch.argmax(outputs.logits[0, -1, :]).item()
        assert ex["prediction"].startswith(tokenizer.decode(token_pred)), ex["id"]

        for layer in range(total_layers):
            for k in args.hook_modules:
                out = model.activations_[f"{k}_{layer}"][0]
                if layer_norm is not None:
                    out = layer_norm(out).detach()
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
        if args.store_topk and len(records) > 200 * total_layers * len(
            args.hook_modules
        ):
            df = pd.DataFrame(records)
            filename = os.path.join(
                args.output_folder,
                "tmp",
                f"extraction_events_{len(tmp_stored_records)}.csv",
            )
            df.to_csv(filename, index=False)
            tmp_stored_records.append(filename)
            records = []

    if args.store_topk:
        filename = os.path.join(
            args.output_folder,
            "tmp",
            f"extraction_events_{len(tmp_stored_records)}.csv",
        )
        pd.DataFrame(records).to_csv(filename, index=False)
        tmp_stored_records.append(filename)
        df = pd.concat([pd.read_csv(f) for f in tmp_stored_records])
        shutil.rmtree(os.path.join(args.output_folder, "tmp"))
    else:
        df = pd.DataFrame(records)
    df.to_csv(os.path.join(args.output_folder, "extraction_events.csv"), index=False)
    with open(os.path.join(args.output_folder, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)
    df[["proj_vec", "language", "relation", "layer", "pred_in_top_1"]].groupby(
        by=["proj_vec", "language", "relation", "layer"], as_index=False
    ).mean().to_csv(
        os.path.join(args.output_folder, "extraction_events_avg_relation.csv"),
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--model_name", type=str)
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
    parser.add_argument("--apply_layer_norm", action="store_true")
    parser.add_argument("--only_subset", action="store_true")
    parser.add_argument("--filter_trivial", action="store_true")
    parser.add_argument("--keep_only_trivial", action="store_true")
    parser.add_argument("--resample_trivial", action="store_true")
    parser.add_argument("--hook_modules", nargs="+")
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
