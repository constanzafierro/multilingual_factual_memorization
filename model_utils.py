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
