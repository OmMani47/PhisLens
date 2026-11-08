"""Train the PhishLens classifier and write the held-out metrics.

From the repository root:

    python -m src.train
    python -m src.train --regenerate

The fit uses a stratified 75/25 split of the synthetic mailbox. Numbers
printed here are not real-world accuracy. See the README.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.dataset import generate_emails, write_csv
from src.paths import DATA_PATH, METRICS_PATH, MODEL_PATH
from src.pipeline import build_pipeline

TEXT_COLUMNS = ["sender", "reply_to", "subject", "body"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PhishLens teaching classifier.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Rewrite the synthetic CSV with the fixed seed before training.",
    )
    return parser.parse_args()


def _records(frame: pd.DataFrame) -> list[dict]:
    return frame[TEXT_COLUMNS].to_dict(orient="records")


def _signed_coefficients(pipeline) -> tuple[list[str], list[float]]:
    names = [str(name) for name in pipeline.named_steps["features"].get_feature_names_out()]
    coefficients = pipeline.named_steps["clf"].coef_.ravel().tolist()
    classes = list(pipeline.named_steps["clf"].classes_)
    if classes[1] != 1:
        coefficients = [-value for value in coefficients]
    return names, coefficients


def _top_terms(names: list[str], coefficients: list[float], prefix: str, limit: int) -> dict:
    paired = [
        (name.split("__")[-1], float(coef))
        for name, coef in zip(names, coefficients)
        if name.startswith(prefix)
    ]
    paired.sort(key=lambda item: item[1], reverse=True)
    return {
        "toward_phishing": [{"term": term, "coefficient": round(coef, 4)} for term, coef in paired[:limit]],
        "toward_legitimate": [
            {"term": term, "coefficient": round(coef, 4)} for term, coef in paired[-limit:][::-1]
        ],
    }


def train(args: argparse.Namespace) -> dict:
    if args.regenerate or not args.data.exists():
        rows = generate_emails(seed=args.seed)
        write_csv(args.data, rows)
        print(f"Wrote {len(rows)} synthetic emails to {args.data}")

    frame = pd.read_csv(args.data, keep_default_na=False)
    missing = [column for column in TEXT_COLUMNS + ["label"] if column not in frame.columns]
    if missing:
        raise SystemExit(f"CSV is missing columns: {', '.join(missing)}")

    labels = frame["label"].astype(int)
    if sorted(labels.unique().tolist()) != [0, 1]:
        raise SystemExit("Labels must be 0 (legitimate) and 1 (phishing).")

    # email_id and category stay out of the model so it cannot read the answer key.
    train_frame, test_frame = train_test_split(
        frame,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )
    pipeline = build_pipeline()
    pipeline.fit(_records(train_frame), train_frame["label"].astype(int))

    y_test = test_frame["label"].astype(int)
    predictions = pipeline.predict(_records(test_frame))
    probabilities = pipeline.predict_proba(_records(test_frame))
    classes = list(pipeline.named_steps["clf"].classes_)
    phishing_index = classes.index(1)

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["legitimate", "phishing"],
        output_dict=True,
        zero_division=0,
    )
    names, coefficients = _signed_coefficients(pipeline)
    heuristic = [
        {
            "feature": name.split("__", 1)[1],
            "coefficient": round(float(coef), 4),
        }
        for name, coef in zip(names, coefficients)
        if name.startswith("heur__")
    ]
    heuristic.sort(key=lambda item: item["coefficient"], reverse=True)

    mistakes = []
    test_reset = test_frame.reset_index(drop=True)
    for row_index, (true_label, predicted, probability) in enumerate(
        zip(y_test.to_numpy(), predictions, probabilities)
    ):
        if int(true_label) == int(predicted):
            continue
        row = test_reset.iloc[row_index]
        mistakes.append(
            {
                "email_id": row["email_id"],
                "category": row["category"],
                "true_label": "phishing" if int(true_label) == 1 else "legitimate",
                "predicted_label": "phishing" if int(predicted) == 1 else "legitimate",
                "phishing_probability": round(float(probability[phishing_index]), 4),
                "subject": row["subject"],
            }
        )

    metrics = {
        "disclaimer": (
            "Scores are from a held-out slice of the synthetic teaching set. "
            "They are not an estimate of accuracy on real mail."
        ),
        "seed": args.seed,
        "test_size": args.test_size,
        "rows": int(len(frame)),
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "positive_class": "phishing",
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision_phishing": round(float(precision_score(y_test, predictions, pos_label=1)), 4),
        "recall_phishing": round(float(recall_score(y_test, predictions, pos_label=1)), 4),
        "f1_phishing": round(float(f1_score(y_test, predictions, pos_label=1)), 4),
        "precision_legitimate": round(float(precision_score(y_test, predictions, pos_label=0)), 4),
        "recall_legitimate": round(float(recall_score(y_test, predictions, pos_label=0)), 4),
        "f1_legitimate": round(float(f1_score(y_test, predictions, pos_label=0)), 4),
        "confusion_matrix": {
            "rows": ["true legitimate", "true phishing"],
            "columns": ["predicted legitimate", "predicted phishing"],
            "matrix": matrix.astype(int).tolist(),
        },
        "classification_report": report,
        "heuristic_coefficients": heuristic,
        "top_word_coefficients": _top_terms(names, coefficients, "word__", 12),
        "misclassified_test_rows": mistakes,
    }

    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model, compress=3)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def _print_metrics(metrics: dict) -> None:
    matrix = metrics["confusion_matrix"]["matrix"]
    print()
    print("PhishLens training")
    print(f"rows: {metrics['rows']}  train: {metrics['train_rows']}  test: {metrics['test_rows']}")
    print("split: stratified, test_size="
          f"{metrics['test_size']}, seed={metrics['seed']}")
    print()
    print("Phishing class on the held-out synthetic test set")
    print(f"  precision  {metrics['precision_phishing']:.3f}")
    print(f"  recall     {metrics['recall_phishing']:.3f}")
    print(f"  f1         {metrics['f1_phishing']:.3f}")
    print(f"  accuracy   {metrics['accuracy']:.3f}")
    print()
    print("Confusion matrix (rows are truth, columns are predictions)")
    print(f"                 legitimate   phishing")
    print(f"  legitimate     {matrix[0][0]:10d}   {matrix[0][1]:8d}")
    print(f"  phishing       {matrix[1][0]:10d}   {matrix[1][1]:8d}")
    print()
    print("Largest security-feature weights (positive leans phishing)")
    for item in metrics["heuristic_coefficients"][:8]:
        print(f"  {item['coefficient']:+.3f}  {item['feature']}")
    if metrics["misclassified_test_rows"]:
        print()
        print(f"Misclassified test rows: {len(metrics['misclassified_test_rows'])}")
        for mistake in metrics["misclassified_test_rows"][:8]:
            print(
                f"  {mistake['email_id']} true={mistake['true_label']} "
                f"pred={mistake['predicted_label']} "
                f"p={mistake['phishing_probability']:.2f} {mistake['subject']}"
            )
    else:
        print()
        print("The synthetic test split had no mistakes. That is not evidence about real inboxes.")
    print()
    print(metrics["disclaimer"])


def main() -> None:
    args = _parse_args()
    metrics = train(args)
    _print_metrics(metrics)
    print(f"Saved model to {args.model}")
    print(f"Saved metrics to {args.metrics}")


if __name__ == "__main__":
    main()
