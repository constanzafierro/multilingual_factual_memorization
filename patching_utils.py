import argparse
import collections
import os
from collections import defaultdict

import numpy as np
import torch
from dataset.data_utils import get_memorized_dataset, find_token_range
from inference.run_inference import prepare_prompt
from third_party.rome.experiments.causal_trace import (
    collect_embedding_std,
    decode_tokens,
    layername,
    plot_trace_heatmap,
    predict_from_input,
    predict_token,
)
from model_utils import load_model_and_tok
from third_party.rome.util import nethook
from transformers import T5TokenizerFast


def trace_with_patch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    trace_layers=None,  # List of traced outputs to return
):
    """The first example in the batch is used for patching in all the subsequent examples.
    The probability is tracked over the mean of all the subsequent examples.
    """
    prng = np.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    embed_layername = layername(model, 0, "embed")
    if "decoder_input_ids" in inp:
        # The encoder and decoder share the embedding layer so we need to check
        # whether we have the encoder input to perform the noise addition.
        encoder_input_shape = inp["input_ids"].shape[-1]

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == embed_layername and (
            "decoder_input_ids" not in inp or x.shape[1] == encoder_input_shape
        ):
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
            if isinstance(t, tuple):
                h[1:, t[1]] = h[0, t[0]]
            else:
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
    if noise:
        probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[
            answers_t
        ]
    else:
        probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1)[:, answers_t]

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced

    return probs


def trace_important_states(
    model, num_layers, inp, e_range, answer_t, noise=0.1, ntoks=None
):
    """Copy of the function in causal_trace.ipynb"""
    table = []
    for ids_key, stack in [("input_ids", "encoder"), ("decoder_input_ids", "decoder")]:
        if ids_key not in inp:
            continue
        if ntoks is None:
            ntoks = inp[ids_key].shape[1]
        for tnum in range(ntoks):
            row = []
            for layer in range(0, num_layers):
                r = trace_with_patch(
                    model,
                    inp,
                    [(tnum, layername(model, stack=stack, num=layer))],
                    answer_t,
                    tokens_to_mix=e_range,
                    noise=noise,
                )
                row.append(r)
            table.append(torch.stack(row))
    return torch.stack(table)


def trace_important_window(
    model, num_layers, inp, e_range, answer_t, kind, window=10, noise=0.1, ntoks=None
):
    """Copy of the function in causal_trace.ipynb"""
    table = []
    for ids_key, stack in [("input_ids", "encoder"), ("decoder_input_ids", "decoder")]:
        if ids_key not in inp:
            continue
        if ntoks is None:
            ntoks = inp[ids_key].shape[1]
        for tnum in range(ntoks):
            row = []
            for layer in range(0, num_layers):
                layerlist = [
                    (tnum, layername(model, stack=stack, num=L, kind=kind))
                    for L in range(
                        max(0, layer - window // 2),
                        min(num_layers, layer - (-window // 2)),
                    )
                ]
                r = trace_with_patch(
                    model, inp, layerlist, answer_t, tokens_to_mix=e_range, noise=noise
                )
                row.append(r)
            table.append(torch.stack(row))
    return torch.stack(table)