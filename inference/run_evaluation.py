from collections import defaultdict
import argparse
import os
import json
import wandb
from datasets import load_dataset
from inference.f1_score import compute_score

from dataset.pararel_utils import OBJECT_KEY


def evaluate(dataset, id_to_prediction, langs):
    # compute F1 as max across any alias for any answer for the most recent, most frequent, or specific-year answer
    qa_targets, qa_predictions = [], []
    num_empty = 0
    if langs:
        langs = set(langs)
        dataset = dataset.filter(lambda ex: ex["language"] in langs)
    for example in dataset:
        query_id = example["id"]
        targets = example[OBJECT_KEY]
        targets = targets if isinstance(targets, list) else [targets]
        prediction = id_to_prediction[query_id]

        if not len(prediction["answer"]):
            num_empty += 1
            continue

        qa_targets.append(
            {
                "answers": {"answer_start": [0] * len(targets), "text": targets},
                "id": query_id,
            }
        )
        qa_predictions.append({"prediction_text": prediction["answer"], "id": query_id})

    print("Evaluating on {} datapoints".format(len(qa_targets)))
    print("Num empty", num_empty)
    if wandb.run is not None:
        wandb.run.summary["empty_preds"] = num_empty
        wandb.run.summary["datapoints"] = len(qa_targets)
    df, scores = compute_score(predictions=qa_predictions, references=qa_targets)
    return df, {"n_datapoints": len(qa_targets), **scores}


def load_predictions(data_path):
    id_to_preds = {}
    with open(os.path.join(data_path, "predictions.json")) as fhandle:
        for line in fhandle:
            data = json.loads(line)
            example_id = data["example_id"]
            data.pop("example_id")
            # We assume there is only one prediction.
            data.update(data["predictions"][0])
            data.pop("predictions")
            id_to_preds[example_id] = data
    return id_to_preds


def compute_metrics(df):
    metrics = defaultdict(float)
    df["language"] = df.apply(lambda ex: ex["id"].split("_")[0], axis=1)
    df["relation"] = df.apply(lambda ex: ex["id"].split("_")[1], axis=1)
    df["template_id"] = df.apply(lambda ex: ex["id"].split("_")[-1], axis=1)
    df["subj_id"] = df.apply(lambda ex: ex["id"].split("_")[-2], axis=1)
    # Count memorized examples per relation, only one template per subject.
    memorized = (
        df[df.exact_match][["relation", "subj_id", "template_id"]]
        .groupby(by=["relation", "subj_id"], as_index=False)
        .agg(list)
    )
    relation_count = (
        memorized[["relation", "subj_id"]]
        .groupby(by=["relation"], as_index=False)
        .count()
        .values
    )
    metrics["memorized_examples"] = len(memorized)
    for relation, count in relation_count:
        metrics[f"memorized_examples/{relation}"] = count
    return metrics


def main(args):
    experiment_dir = os.path.join(
        args.output_dir, os.path.basename(args.prediction_path)
    )
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    dataset = load_dataset(args.dataset_name)["train"]
    id_to_prediction = load_predictions(args.predictions_path)
    df, scores = evaluate(dataset, id_to_prediction, langs=args.langs)
    wandb.log({k: v for k, v in scores.items() if not isinstance(v, list)})
    df.to_json(os.path.join(experiment_dir, "eval_per_example.json"))
    wandb.log(compute_metrics(df))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="coastalcph/xlingual_mpararel",
        help="",
    )
    parser.add_argument("--predictions_path", type=str, help="Path to predictions")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Dir where model outputs will be stored",
    )
    parser.add_argument("--exp_name", type=str, help="Experiment name")
    parser.add_argument("--langs", default=[], nargs="+", help="Experiment name")
    args = parser.parse_args()

    wandb.init(
        project="xlingual_mpararel_eval",
        name=args.exp_name,
        config=args,
    )

    main(args)
