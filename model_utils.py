from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LlamaForCausalLM,
    AutoConfig,
)
import torch
from third_party.rome.experiments.causal_trace import ModelAndTokenizer


def load_model_and_tok(model_name_or_path, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "t5" in model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            load_in_8bit="xxl" in model_name_or_path,
            device_map="auto",
        )
    elif "polylm" in model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            device_map="auto",
            load_in_8bit="13b" in model_name_or_path,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return ModelAndTokenizer(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        num_layers=config.num_layers,
    )


def load_tokenizer(model_name_or_path):
    tokenizer_args = {}
    if "alpaca" in model_name_or_path or "llama" in model_name_or_path.lower():
        # the fact tokenizer causes issues with protobuf and tokenizers libraries
        tokenizer_args = {"use_fast": False}
    elif "polylm" in model_name_or_path:
        tokenizer_args = {"legacy": False, "use_fast": False}
    return AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_args)
