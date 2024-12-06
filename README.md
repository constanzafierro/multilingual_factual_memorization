# Crosslingual Factual Memorization

Code for the paper ["How Do Multilingual Language Models Remember? Investigating Multilingual Factual Recall Mechanisms"](https://arxiv.org/abs/2410.14387)

## Data
- https://github.com/coastalcph/mpararel augmented with aliases (see `dataset/get_aliases.py`)
- Prepared the data and pushed to HuggingFace:
    - [coastalcph/mpararel_autorr](https://huggingface.co/datasets/coastalcph/mpararel_autorr): only templates with the object at the end.
    - [coastalcph/xlingual_mpararel_autorr](https://huggingface.co/datasets/coastalcph/xlingual_mpararel_autorr): only templates with the object at the end, only subset of subjects available in all languages.
    - [coastalcph/xlingual_mpararel_mlm](https://huggingface.co/datasets/coastalcph/xlingual_mpararel_mlm): all templates, only subset of subjects available in all languages.

## Knowledge Evaluation
`inference/run_inference.py`: runs inference on a HF dataset, stores predictions and logits in a file.

`inference/run_evaluation.py`: computes f1, exact match, and number of memorized examples.

## Experiments
To run any of the experiments you have to have run the knowledge evaluation first, so that the subset of memorized examples can be selected.

`causal_analysis.py`: experiments of section 3

`attention_knockout.py`: experiments of section 4 (Information Flow paragraph)

`extraction_rate.py`: experiments of section 4 (Prediction Extraction paragraph)

`patching_experiments.py`: experiments of section 5

## Citation
If you use this code, please consider citing:
```latex
@article{fierro2024multilingual,
  title={How Do Multilingual Models Remember? Investigating Multilingual Factual Recall Mechanisms},
  author={Fierro, Constanza and Foroutan, Negar and Elliott, Desmond and S{\o}gaard, Anders},
  journal={arXiv preprint arXiv:2410.14387},
  year={2024}
}
```
