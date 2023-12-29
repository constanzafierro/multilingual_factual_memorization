import collections
import itertools
import json
from transformers import (
    LlamaTokenizerFast,
    GPT2TokenizerFast,
    T5TokenizerFast,
    LlamaTokenizer,
    GPTNeoXTokenizerFast,
)
import re
import os
import numpy as np
from tqdm import tqdm

import wandb

OBJECT_KEY = "obj_label"
OBJECT_QCODE = "obj_uri"
SUBJECT_KEY = "sub_label"
SUBJECT_QCODE = "sub_uri"
PATTERN_KEY = "pattern"
MPARAREL_FOLDER = "mpararel/data/mpararel_reviewed"
PATTERNS_FOLDER = "patterns"
TUPLES_FOLDER = "tuples"

TOKENIZER_TO_PREPEND_SPACE = {
    LlamaTokenizerFast: False,
    GPT2TokenizerFast: True,
    T5TokenizerFast: True,
    LlamaTokenizer: False,
    GPTNeoXTokenizerFast: True,
}


def _get_mpararel_items(path_to_file, key_item=None):
    items = []
    if not os.path.exists(path_to_file):
        return items
    with open(path_to_file) as f:
        for line in f:
            data = json.loads(line)
            if data:
                if key_item:
                    items.append(data[key_item])
                else:
                    items.append(data)
    return items


def clean_template(template):
    template = template.lower()
    # Remove extra spaces and extra brackets from the subject/object and
    # capitalize them.
    template = re.sub("\[+ ?[x] ?\]+", "[X]", template)
    template = re.sub("\[+ ?[y] ?\]+", "[Y]", template)
    # Remove final puntuaction
    template = re.sub(r"[.:。]", "", template)
    # Remove extra spaces
    template = re.sub(r" +", " ", template)
    template = re.sub(r" $", "", template)
    return template


def get_mpararel_templates(lang, relation_filename, mask_lm=False):
    templates = _get_mpararel_items(
        os.path.join(MPARAREL_FOLDER, PATTERNS_FOLDER, lang, relation_filename),
        PATTERN_KEY,
    )
    templates = [clean_template(t) for t in templates]
    if mask_lm:
        return templates
    final_templates = []
    for template in templates:
        template_object_removed = re.sub(r" \[Y\]\s?\.?$", "", template.strip()).strip()
        if "[Y]" in template_object_removed:
            continue
        final_templates.append(template)
    if wandb.run is not None:
        wandb.run.summary[f"templates_count/{lang}_{relation_filename}"] = len(
            relation_filename
        )
    return final_templates


def get_mpararel_subject_object(
    lang, relation_filename, tuples_folder=TUPLES_FOLDER, only_tuple=True
):
    tuples_data = [
        e
        for e in _get_mpararel_items(
            os.path.join(MPARAREL_FOLDER, tuples_folder, lang, relation_filename)
        )
    ]
    counts = collections.Counter([data[SUBJECT_KEY] for data in tuples_data])
    unique_subjects = []
    for data in tuples_data:
        if counts[data[SUBJECT_KEY]] > 1:
            continue
        out = (data[SUBJECT_KEY], data[OBJECT_KEY]) if only_tuple else data
        unique_subjects.append(out)
    return unique_subjects


def get_target_object(tokenizer, object_):
    # This works for SentencePiece tokenizers (e.g. T5) which
    # tokenize space and no space to the same id. It's also what
    # GPT-2 BPE tokenizer expects for words not at the beginning of
    # the sentence.
    if TOKENIZER_TO_PREPEND_SPACE[type(tokenizer)]:
        target_object = " " + object_.strip()
    else:
        if not isinstance(tokenizer, LlamaTokenizerFast) and not isinstance(
            tokenizer, LlamaTokenizer
        ):
            raise Exception(
                "Check that no space in the template and target "
                "with no space is the right setup for tokenizer {}".format(
                    type(tokenizer)
                )
            )
        target_object = object_.strip()
    return target_object


class PararelPrompt:
    def __init__(self, prompt, subject, target_object, target_token_id):
        self.prompt = prompt.strip()
        self.subject = subject
        self.target_object = target_object
        self.target_token_id = target_token_id
        self.model_input = self.prompt.replace("[X]", self.subject)
