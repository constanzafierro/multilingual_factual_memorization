# Crosslingual Factual Memorization

HF space: https://huggingface.co/spaces/coastalcph/xlingual_memorization

## Data
- https://github.com/coastalcph/mpararel augmented with aliases (see `dataset/get_aliases.py`)
- Prepared the data and pushed to HuggingFace:
    - [coastalcph/mpararel_autorr](https://huggingface.co/datasets/coastalcph/mpararel_autorr): only templates with the object at the end.
    - [coastalcph/xlingual_mpararel_autorr](https://huggingface.co/datasets/coastalcph/xlingual_mpararel_autorr): only templates with the object at the end, only subset of subjects available in all languages.
    - [coastalcph/xlingual_mpararel_mlm](https://huggingface.co/datasets/coastalcph/xlingual_mpararel_mlm): all templates, only subset of subjects available in all languages.

## Evals
`inference/run_inference.py`: runs inference on a HF dataset, stores predictions and logits in a file.
`inference/run_evaluation.py`: computes f1, exact match, and number of memorized examples.
