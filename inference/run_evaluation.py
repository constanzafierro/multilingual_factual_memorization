from collections import defaultdict
import argparse
import os
import json
import wandb
from datasets import load_dataset
from f1_score import compute_score

from dataset.pararel_utils import OBJECT_KEY


def evaluate(dataset, id_to_prediction):
    # compute F1 as max across any alias for any answer for the most recent, most frequent, or specific-year answer
    qa_targets, qa_predictions = [], []
    num_empty = 0
    for example in dataset:
        query_id = example["id"]
        targets = example[OBJECT_KEY]
        targets = targets if isinstance(targets, list) else [targets]
        prediction = id_to_prediction[query_id]

        if not len(prediction["answer"]):
            num_empty += 1
            # print("Warning: the prediction for query='{}' was empty.".format(query))
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
    df, scores = compute_score(predictions=qa_predictions, references=qa_targets)
    return df, {"n_datapoints": len(qa_targets), **scores}


def load_queries(data_path):
    unique_queries = dict()
    queries = load_dataset(data_path, split="train")
    for query in queries:
        query_id = "_".join(query["id"].split("_")[:2])
        if query_id not in unique_queries and len(query["answer"]):
            unique_queries[query_id] = query
    return unique_queries


def load_aliases(data_path):
    all_aliases = dict()
    aliases = load_dataset(data_path, split="train")
    for qid, al in aliases[0].items():
        all_aliases[qid] = al
    return all_aliases


def load_predictions(data_path):
    predictions = {}
    with open(data_path) as fhandle:
        for line in fhandle:
            data = json.loads(line)
            example_id = data["example_id"]
            del data["example_id"]
            data["predictions"] = [p for p in data["predictions"] if len(p["answer"])]
            if len(predictions) == 0:
                print(
                    "Example of data predictions for example_id={}: {}".format(
                        example_id, data
                    )
                )
            predictions[example_id] = data

    return predictions


def compute_metrics(df):
    # columns=["id", "prediction", "ground_truth", "f1", "exact_match"]
    metrics = defaultdict(float)
    # TODO: split id in {lang}_{relation}_{tuple_[SUBJECT_QCODE]}_{template_id}
    df.groupby(by=[])
    return metrics


def main(args):
    experiment_dir = os.path.join(args.output_dir, args.exp_name)
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)

    dataset = load_dataset(args.dataset_name)["train"]
    id_to_prediction = load_predictions(args.predictions_path)
    df, scores = evaluate(dataset, id_to_prediction)
    wandb.log({k: v for k, v in scores.items() if not isinstance(v, list)})
    df.to_json(os.path.join(experiment_dir, "eval_per_example.json"))
    wandb.log(compute_metrics(df))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="coastalcph/mpararel",
        help="",
    )
    parser.add_argument("--predictions_path", type=str, help="Path to predictions")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Dir where model outputs will be stored",
    )
    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
    args = parser.parse_args()

    wandb.init(
        project="mpararel_eval",
        name=args.exp_name,
        config=args,
    )

    main(args)
