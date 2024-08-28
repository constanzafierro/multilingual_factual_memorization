from collections import defaultdict
import argparse
import os
import json
import wandb
from datasets import load_dataset
from inference.f1_score import compute_score
import re
import collections
from glob import glob
from dataset.pararel_utils import OBJECT_KEY
from dataset.data_utils import log_trivial_examples_counts, get_dataset_name


def evaluate(dataset, id_to_prediction, language):
    # compute F1 as max across any alias for any answer for the most recent, most frequent, or specific-year answer
    qa_targets, qa_predictions = [], []
    counts = collections.defaultdict(int)
    dataset = dataset.filter(lambda ex: ex["language"] == language)
    for example in dataset:
        query_id = example["id"]
        if query_id not in id_to_prediction:
            # TODO: There should be no missing ids. Need to run inference on these.
            counts["missing_ids"] += 1
            continue

        targets = example[OBJECT_KEY]
        targets = targets if isinstance(targets, list) else [targets]
        prediction = id_to_prediction[query_id]

        if not len(prediction):
            counts["empty_preds"] += 1
            continue

        qa_targets.append(
            {
                "answers": {"answer_start": [0] * len(targets), "text": targets},
                "id": query_id,
            }
        )
        qa_predictions.append({"prediction_text": prediction, "id": query_id})

    print("Evaluating on {} datapoints".format(len(qa_targets)))
    print("Num empty", counts["empty_preds"])
    df, scores = compute_score(predictions=qa_predictions, references=qa_targets)
    return df, {"n_datapoints": len(qa_targets), **counts, **scores}


def load_predictions(data_path):
    id_to_preds = {}
    with open(os.path.join(data_path, "predictions.json")) as fhandle:
        for line in fhandle:
            data = json.loads(line)
            # We assume there is only one prediction.
            id_to_preds[data["example_id"]] = data["predictions"][0]["answer"]
    return id_to_preds


def load_sentinel_prediction(data_path):
    id_to_preds = {}
    with open(os.path.join(data_path, "raw_predictions.json")) as fhandle:
        for line in fhandle:
            data = json.loads(line)
            answer = data["predictions"][0]["answer"]
            answer = answer.replace("<pad>", "").replace("</s>", "").strip()
            if not re.match(r"(<extra_id_\d>.*)+", answer):
                raise Exception(
                    "'{}' did not match the regex with sentinel tokens.".format(answer)
                )
            re_split = re.split(r"<extra_id_(\d)>", answer)
            sentinel_index, pred = re_split[1], re_split[2]
            assert sentinel_index == "0", answer
            id_to_preds[data["example_id"]] = pred
    return id_to_preds


def add_raw_prediction(df, predictions_path, decoder_key):
    id_to_preds = {}
    with open(os.path.join(predictions_path, "raw_predictions.json")) as fhandle:
        for line in fhandle:
            data = json.loads(line)
            id_to_preds[data["example_id"]] = data["predictions"][0]["answer"]
    key = (
        "decoder_pred_with_special_tokens"
        if decoder_key
        else "raw_pred_with_special_tokens"
    )
    df[key] = df.apply(lambda row: id_to_preds[row["id"]], axis=1)
    return df


def compute_metrics(df):
    metrics = defaultdict(float)
    df["language"] = df.apply(lambda ex: ex["id"].split("_")[0], axis=1)
    df["relation"] = df.apply(lambda ex: ex["id"].split("_")[1], axis=1)
    df["template_id"] = df.apply(lambda ex: ex["id"].split("_")[-1], axis=1)
    df["subj_id"] = df.apply(lambda ex: ex["id"].split("_")[-2], axis=1)
    if len(df[df.exact_match]) == 0:
        metrics["memorized_examples"] = 0
        metrics["memorized_relations"] = 0
        return {}
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
    metrics["memorized_relations"] = len(relation_count)
    metrics["total_relations"] = len(df.relation.unique())
    for relation, count in relation_count:
        metrics[f"memorized_examples/{relation}"] = count
    return metrics


def main(args):
    dataset_name = get_dataset_name(args.model_name, args.language)
    pattern = os.path.join(
        args.predictions_folder,
        "".join(
            [
                args.language,
                "{}",
                dataset_name.split("/")[1],
                "--",
                args.model_name.replace("/", "__"),
            ]
        ),
    )
    predictions_path = pattern.format("--")
    if not os.path.isfile(os.path.join(predictions_path, "predictions.json")):
        predictions_path = pattern.format("_")
    experiment_dir = os.path.join(args.output_dir, os.path.basename(predictions_path))
    if args.use_sentinel_prediction:
        experiment_dir = os.path.join(experiment_dir, "sentinel_pred")
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    wandb.config["final_dir"] = experiment_dir
    wandb.config["predictions_path"] = predictions_path

    dataset = load_dataset(dataset_name)["train"]
    if args.use_sentinel_prediction:
        id_to_prediction = load_sentinel_prediction(predictions_path)
        wandb.run.name += " sentinel_prediction"
    else:
        id_to_prediction = load_predictions(predictions_path)
    df, scores = evaluate(dataset, id_to_prediction, args.language)
    df = add_raw_prediction(
        df, predictions_path, decoder_key=args.use_sentinel_prediction
    )
    wandb.log({k: v for k, v in scores.items() if not isinstance(v, list)})
    df.to_json(
        os.path.join(experiment_dir, "eval_per_example_records.json"), orient="records"
    )
    log_trivial_examples_counts(df, dataset)
    wandb.log(compute_metrics(df))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--language", type=str)
    parser.add_argument("--predictions_folder", type=str, help="Path to predictions")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Dir where model outputs will be stored",
    )
    parser.add_argument("--use_sentinel_prediction", action="store_true")
    args = parser.parse_args()

    wandb.init(
        project="xlingual_mpararel_eval",
        name=os.path.basename(" ".join(args.model_name, args.language)),
        config=args,
    )

    main(args)
