from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
)
import torch
from third_party.rome.experiments.causal_trace import ModelAndTokenizer


def load_model_and_tok(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ""
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=not isinstance(model, LlamaForCausalLM),
    )
    return ModelAndTokenizer(
        model_name=args.model_name, model=model, tokenizer=tokenizer
    )


def load_tokenizer(model_name_or_path):
    tokenizer_args = {}
    if "alpaca" in model_name_or_path or "llama" in model_name_or_path.lower():
        # the fact tokenizer causes issues with protobuf and tokenizers libraries
        tokenizer_args = {"use_fast": False}
    elif "polylm" in model_name_or_path:
        tokenizer_args = {"legacy": False, "use_fast": False}
    return AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_args)
