import argparse
import functools
import json
import os
from functools import partial

import pandas as pd
import torch
import wandb
from tqdm import tqdm

from dataset.data_utils import find_token_range, get_dataset_name, get_memorized_dataset
from model_utils import load_model_and_tok
from third_party.rome.experiments.causal_trace import (
    decode_tokens,
    layername,
    predict_from_input,
)
from third_party.rome.util.nethook import get_module


def remove_wrapper(model, hooks):
    for i, hook in hooks:
        module_to_hook = get_module(model, layername(model, num=i, kind="attn"))
        module_to_hook.forward = hook


def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()


# To block attention edges, we zero-out entries in the attention mask.
# To do this, we add a wrapper around the attention module, because
# the mask is passed as an additional argument, which could not be fetched
# with standard hooks before pytorch 2.0.
def set_block_attn_hooks(model, attn_layer_to_blockage):

    def wrap_attn_forward(forward_fn, model_, from_to_index_):
        @functools.wraps(forward_fn)
        def wrapper_fn(*args, **kwargs):
            new_args = []
            new_kwargs = {}
            for arg in args:
                new_args.append(arg)
            for k, v in kwargs.items():
                new_kwargs[k] = v

            hs = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            num_hs_tokens = list(hs[0].size())[0]
            # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
            attention_mask_key = (
                "attention_mask" if "attention_mask" in kwargs else "mask"
            )
            causal_attention = (
                kwargs[attention_mask_key].shape[-2]
                == kwargs[attention_mask_key].shape[-1]
            )
            # The mask argument is only used in the first layer in mt5,
            # afterwards the position_bias term drives the masking logic.
            attention_mask_key = (
                "attention_mask" if "attention_mask" in kwargs else "position_bias"
            )
            # Decoder only model or the decoder self attention block.
            if causal_attention:
                attn_mask = torch.tril(
                    torch.ones((num_hs_tokens, num_hs_tokens), dtype=torch.uint8)
                )
                for s, t in from_to_index_:
                    attn_mask[s, t] = 0
                attn_mask = attn_mask.repeat(1, kwargs["attention_mask"].shape[1], 1, 1)
                attn_mask = attn_mask.to(dtype=model_.dtype)  # fp16 compatibility
                # Padding elements are indicated by very large negative values.
                attn_mask = (1.0 - attn_mask) * torch.finfo(model_.dtype).min
            # Encoder self attention or the decoder cross attention block.
            else:
                # The position_bias is (batch_size, num_heads, hidden_states_length, key_values_length)
                attn_mask = kwargs[attention_mask_key].copy()
                for s, t in from_to_index_:
                    attn_mask[:, :, s, t] = torch.finfo(model_.dtype).min

            attn_mask = attn_mask.to(hs.device)
            new_kwargs[attention_mask_key] = attn_mask

            return forward_fn(*new_args, **new_kwargs)

        return wrapper_fn

    hooks = []
    for (stack, layer, kind), blockage in attn_layer_to_blockage.items():
        module_to_hook = get_module(
            model, layername(model, num=layer, stack=stack, kind=kind)
        )
        hook = module_to_hook.forward
        module_to_hook.forward = wrap_attn_forward(
            module_to_hook.forward,
            model,
            blockage,
        )
        hooks.append((layer, hook))

    return hooks


def trace_with_attn_blockage(
    model,
    inp,
    layer_to_source_target_blockage,  # A list of (source index, target index) to block
    answers_t,
):
    """Forward pass with source attn being blocked to the target."""
    with torch.no_grad():
        # set hooks
        block_attn_hooks = set_block_attn_hooks(model, layer_to_source_target_blockage)

        # get prediction
        outputs_exp = model(**inp)

        # remove hooks
        remove_wrapper(model, block_attn_hooks)

    probs = torch.softmax(outputs_exp.logits[0, -1, :], dim=0)[answers_t]

    return probs


def get_output_dir(args):
    data_id = args.language
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
    return output_folder


def get_block_indices(model_name, e_range, inp):
    def get_block_config(
        block_from, block_indices, layer, total_layers, stack="encoder", kind="attn"
    ):
        layers_to_block = range(
            max(0, layer - args.patch_k_layers // 2),
            min(total_layers, layer - (-args.patch_k_layers // 2)),
        )
        return {
            (stack, l_block, kind): [
                (block_from, to_block) for to_block in block_indices
            ]
            for l_block in layers_to_block
        }

    input_ids_count = inp["input_ids"].shape[1]
    if "t5" not in model_name:
        last_token = input_ids_count - 1
        block_indices_desc = [
            ([x for x in e_range], "subject"),
            ([e_range[-1]], "last_subject"),
            (
                [x for x in range(input_ids_count - 1) if x not in e_range],
                "non-subject",
            ),
            ([last_token], "last"),
        ]
        return [
            (
                partial(
                    get_block_config,
                    block_from=last_token,
                    block_indices=block_indices,
                ),
                block_desc,
            )
            for block_indices, block_desc in block_indices_desc
        ]
    else:
        decoder_ids_count = inp["decoder_input_ids"].shape[1]
        last_token = decoder_ids_count - 1
        sentinel_token = (inp["input_ids"][0] == 250099).nonzero().item()
        block_indices_desc = [
            (
                sentinel_token,
                [x for x in e_range],
                "encoder",
                "attn",
                "sentinel->subject",
            ),
            (
                sentinel_token,
                [x for x in range(input_ids_count - 1) if x not in e_range],
                "encoder",
                "attn",
                "sentinel->non_subject",
            ),
            (sentinel_token, [sentinel_token], "encoder", "attn", "sentinel->itself"),
            (
                last_token,
                [x for x in e_range],
                "decoder",
                "cross_attn",
                "last->subject",
            ),
            (
                last_token,
                [x for x in range(input_ids_count - 1) if x not in e_range],
                "decoder",
                "cross_attn",
                "last->non_subject",
            ),
            (
                last_token,
                [i for i in range(last_token)],
                "decoder",
                "attn",
                "last->decoder_tokens",
            ),
            (last_token, [last_token], "decoder", "attn", "last->itself"),
        ]
        return [
            (
                partial(
                    get_block_config,
                    block_from=block_from,
                    block_indices=block_to_indices,
                    stack=stack,
                    kind=attn_kind,
                ),
                block_desc,
            )
            for block_from, block_to_indices, stack, attn_kind, block_desc in block_indices_desc
        ]


def main(args):
    output_folder = get_output_dir(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    # Run attention knockouts
    results = []
    block_configs = []
    for ex in tqdm(ds, desc="Examples"):
        ex_id = ex["id"]
        input_prompt = ex["query_inference"]
        subject = ex["sub_label"]
        inp = {"input_ids": torch.tensor([ex["input_ids"]]).to(device)}
        inp["attention_mask"] = torch.ones_like(inp["input_ids"])
        if ex["decoder_input_ids"] is not None:
            decoder_input_ids = torch.tensor([ex["decoder_input_ids"]]).to(device)
            inp = {**inp, "decoder_input_ids": decoder_input_ids}

        e_range = find_token_range(
            mt.tokenizer, inp["input_ids"][0], subject, input_prompt
        )
        e_range = list(range(e_range[0], e_range[1]))
        is_subject_position_zero = input_prompt.startswith(subject)

        answer_t, base_score = [d[0] for d in predict_from_input(mt.model, inp)]
        base_score = base_score.cpu().item()
        [answer] = decode_tokens(mt.tokenizer, [answer_t])

        for block_indices, block_desc in get_block_indices(
            args.model_name, e_range, inp
        ):
            for layer in range(mt.num_layers):
                block_config = block_indices(layer=layer, total_layers=mt.num_layers)
                probs = trace_with_attn_blockage(mt.model, inp, block_config, answer_t)
                new_score = probs.cpu().item()
                results.append(
                    {
                        "ex_id": ex_id,
                        "input_prompt": input_prompt,
                        "answer": answer,
                        "block_desc": block_desc,
                        "layer": layer,
                        "base_score": base_score,
                        "new_score": new_score,
                        "relative diff": (new_score - base_score) * 100.0 / base_score,
                        "is_subject_position_zero": is_subject_position_zero,
                    }
                )
                block_configs.append(
                    {
                        "block_desc": block_desc,
                        "layer": layer,
                        "block_config": block_config,
                    }
                )
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_folder, "results.csv"), index=False)
    print("Writing config")
    with open(os.path.join(output_folder, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)
    with open(os.path.join(output_folder, "block_configs.json"), "w") as f:
        json.dump(block_configs, f)


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
        "--language",
        type=str,
        help="",
    )
    parser.add_argument("--only_subset", action="store_true")
    parser.add_argument("--filter_trivial", action="store_true")
    parser.add_argument("--keep_only_trivial", action="store_true")
    parser.add_argument("--resample_trivial", action="store_true")
    parser.add_argument("--patch_k_layers", type=int, default=10)
    args = parser.parse_args()
    if not args.model_name:
        args.model_name = args.model_name_or_path.replace("/", "__")
    wandb.init(
        project="xfact_att_knockout",
        name=" ".join(
            [
                args.model_name,
                args.language,
            ]
        ),
        config=args,
    )
    main(args)
