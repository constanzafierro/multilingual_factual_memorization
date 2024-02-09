import argparse
import json
import os

import numpy as np
import torch
import wandb
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BloomTokenizerFast,
    GenerationConfig,
    LlamaTokenizer,
    LlamaTokenizerFast,
    PreTrainedTokenizerFast,
    T5TokenizerFast,
    XGLMTokenizerFast,
)

from dataset.pararel_utils import SUBJECT_QCODE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_BEAMS = 1


def prepare_prompt(query, model_name_or_path, instruction, is_mlm_template):
    if "mt5" in model_name_or_path:
        if is_mlm_template:
            return query.replace("[Y]", "<extra_id_0>")
        return query + " <extra_id_0> "
    # Assume autorregressive template and model.
    return query


def remove_bos(tokenizer, seq, input_ids):
    def ignore_prompt(seq, input_ids):
        return seq[input_ids.shape[1] :].cpu().tolist()

    tok_to_func = {
        # Ignore the prompt.
        LlamaTokenizer: ignore_prompt,
        LlamaTokenizerFast: ignore_prompt,
        XGLMTokenizerFast: ignore_prompt,
        BloomTokenizerFast: ignore_prompt,
        PreTrainedTokenizerFast: lambda seq, input_ids: seq[input_ids.shape[1] :]
        .cpu()
        .tolist(),
        # Ignore the BOS (pad) token.
        T5TokenizerFast: lambda seq, _: seq.cpu().tolist()[1:],
    }
    return tok_to_func[type(tokenizer)](seq, input_ids)


def get_ids_to_ignore(tokenizer):
    ids_to_ignore = {
        LlamaTokenizer: [tokenizer.bos_token_id, tokenizer.eos_token_id],
        LlamaTokenizerFast: [tokenizer.bos_token_id, tokenizer.eos_token_id],
        # Ignore EOS.
        T5TokenizerFast: [
            tokenizer.eos_token_id,
            *[
                tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f"<extra_id_{i}>"))[
                    0
                ]
                for i in range(100)
            ],
        ],
        # Ignore EOS.
        PreTrainedTokenizerFast: [tokenizer.eos_token_id],
        XGLMTokenizerFast: [tokenizer.eos_token_id],
        BloomTokenizerFast: [],
    }
    return ids_to_ignore[type(tokenizer)]


def get_full_stop(tokenizer):
    # Token id of a full stop when not at the beggining of a word so it could be
    # different than tokenizer.tokens_to_ids(tokenizer.tokenize('.')).
    ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("a."))
    return ids[-1]


def get_scores(model_output, input_ids, prompt, query, tokenizer):
    """Assumes num_beam=1. Gets the token scores for every token that is not BOS, EOS or fullstop,
    gets the first non-the token score and computes pplx."""
    sequence = remove_bos(tokenizer, model_output["sequences"][0], input_ids)
    assert len(sequence) == len(model_output["scores"])
    token_scores = []
    trimmed_sequence = []
    ids_to_ignore = set(get_ids_to_ignore(tokenizer))
    for idx, score in zip(sequence, model_output["scores"]):
        if idx not in ids_to_ignore:
            token_scores.append(torch.softmax(score, 1)[:, idx].cpu().item())
            trimmed_sequence.append(idx)
    if trimmed_sequence and trimmed_sequence[-1] == get_full_stop(tokenizer):
        token_scores = token_scores[:-1]
        trimmed_sequence = trimmed_sequence[:-1]
    answer = tokenizer.decode(trimmed_sequence).strip()
    words = answer.split()
    if (
        not token_scores
        or not words
        or (
            (len(token_scores) == 1 or len(words) == 1)
            and words[0] in ["the", "a", "an"]
        )
    ):
        print(
            "Warning: Empty generation. input_ids={}, output_sequence={}".format(
                input_ids, sequence
            )
        )
        return "", [], 0, float("inf")
    first_token_score = (
        token_scores[1] if words[0] in ["the", "a", "an"] else token_scores[0]
    )
    perplexity = np.exp(-np.mean(np.log(token_scores)))

    return answer, token_scores, first_token_score, perplexity


def get_generation_config(tokenizer):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return GenerationConfig(
        max_new_tokens=50,
        num_beams=NUM_BEAMS,
        do_sample=False,
        output_hidden_states=False,
        output_scores=False,
        num_return_sequences=NUM_BEAMS,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id,
    )


def write_outputs(outputs, experiment_dir):
    print("Writing outputs")
    for key in outputs:
        with open(os.path.join(experiment_dir, key + ".json"), "a") as outfile:
            for item in outputs[key]:
                outfile.write(json.dumps(item))
                outfile.write("\n")


def inference(dataset, tokenizer, model, args, experiment_dir):
    config = get_generation_config(tokenizer)
    outputs = {key: [] for key in ["raw_predictions", "predictions"]}
    for example_index, example in enumerate(tqdm(dataset)):
        example_id = example["id"]
        query = example["query"]
        with torch.no_grad():
            prompt = prepare_prompt(
                query, args.model_name_or_path, args.instruction, args.is_mlm_template
            )
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            model_output = model.generate(
                input_ids, generation_config=config, output_scores=True
            )

        answer, token_scores, first_token_score, perplexity = get_scores(
            model_output, input_ids, prompt, query, tokenizer
        )
        outputs["raw_predictions"].append(
            {
                "example_id": example_id,
                "query": query,
                "predictions": [
                    {
                        "output_ids": model_output["sequences"][0].cpu().tolist(),
                        "answer": tokenizer.decode(model_output["sequences"][0]),
                    }
                ],
            }
        )
        outputs["predictions"].append(
            {
                "example_id": example_id,
                "query": query,
                "predictions": [
                    {
                        "answer": answer,
                        "per_token_probability": token_scores,
                        "first_token_probability": first_token_score,
                        "perplexity": perplexity,
                    }
                ],
            }
        )
        if example_index > 0 and example_index % 2000 == 0:
            write_outputs(outputs, experiment_dir)
            outputs = {key: [] for key in ["raw_predictions", "predictions"]}
    write_outputs(outputs, experiment_dir)


def main(args):
    experiment_name = "{}--{}".format(
        args.exp_name, args.model_name_or_path.replace("/", "-")
    )
    experiment_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    wandb.config["final_dir"] = experiment_dir

    print("Loading model")
    use_fast = True
    if (
        "alpaca" in args.model_name_or_path
        or "llama" in args.model_name_or_path.lower()
    ):
        # the fact tokenizer causes issues with protobuf and tokenizers libraries
        use_fast = False
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=use_fast
    )
    if "t5" not in args.model_name_or_path:
        if args.cache_dir is not None:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path, cache_dir=args.cache_dir
            ).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(
                device
            )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path, load_in_8bit=True, device_map="auto"
        )
    model.eval()

    print("Loading dataset")
    dataset = load_dataset(args.dataset_name)["train"]
    if args.only_languages:
        langs = set(args.only_languages)
        dataset = dataset.filter(lambda ex: ex["language"] in langs)

    if args.topk_popular_subjects != -1:
        popular_subjs = set()
        for relation in os.listdir(args.subjects_count_folder):
            with open(os.path.join(args.subjects_count_folder, relation)) as f:
                qcodes_and_counts = json.load(f)
                popular_subjs.extend(
                    [i for i, _ in qcodes_and_counts[: args.topk_popular_subjects]]
                )
        dataset = dataset.filter(lambda ex: ex[SUBJECT_QCODE] in popular_subjs)

    print("Running inference")
    inference(dataset, tokenizer, model, args, experiment_dir)

    print("Writing config")
    with open(os.path.join(experiment_dir, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="coastalcph/xlingual_mpararel_autorr",
        help="",
    )
    parser.add_argument(
        "--is_mlm_template",
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--only_languages",
        nargs="+",
        default=[],
        help="",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Dir where model outputs will be stored",
    )
    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="huggyllama/llama-7b",
        help="Model name or path",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="",
    )
    parser.add_argument(
        "--topk_popular_subjects",
        type=int,
        default=-1,
        help="",
    )
    parser.add_argument(
        "--subjects_count_folder",
        type=str,
        default=None,
        help="",
    )
    args = parser.parse_args()

    project_name = "xlingual_inference"
    wandb.init(
        project=project_name,
        name=args.exp_name,
        config=args,
    )

    main(args)
