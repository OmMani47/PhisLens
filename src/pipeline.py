"""Sklearn pipeline: security features + word/character TF-IDF + logistic regression.

Logistic regression is the classifier because each weight is a number a
student can read. A positive weight pushes the email toward phishing when
that feature is present. Multiplying the weight by the value in this email
is the feature's contribution to the decision.

Heuristic features are left unscaled on purpose. They are already 0/1 flags
or small capped counts, so a contribution stays in the same units as the
coefficient.
"""

from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src.features import FEATURE_NAMES, Analysis, analyze, record_to_text
from src.paths import MODEL_PATH

_PHISHING = 1


def _records(X) -> list:
    if hasattr(X, "to_dict") and hasattr(X, "columns"):
        return X.to_dict(orient="records")
    return list(X)


class RecordToText(BaseEstimator, TransformerMixin):
    """Turn an email record into the string TF-IDF should read."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [record_to_text(record) for record in _records(X)]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(["email_text"], dtype=object)


class HandcraftedFeatures(BaseEstimator, TransformerMixin):
    """Dense block of interpretable security features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = [analyze(record).vector() for record in _records(X)]
        if not rows:
            return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64)
        return np.asarray(rows, dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURE_NAMES, dtype=object)


def build_pipeline() -> Pipeline:
    """Assemble an unfitted pipeline. ``train.py`` fits and saves it."""
    word = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=4000,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9']+\b",
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=3000,
        sublinear_tf=True,
    )
    features = FeatureUnion(
        [
            ("heur", HandcraftedFeatures()),
            ("word", Pipeline([("text", RecordToText()), ("tfidf", word)])),
            ("char", Pipeline([("text", RecordToText()), ("tfidf", char)])),
        ]
    )
    # liblinear is a strong fit for a linear model of this size and is
    # deterministic given random_state. class_weight keeps a slight imbalance
    # from swallowing the minority class.
    classifier = LogisticRegression(
        C=1.0,
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )
    return Pipeline([("features", features), ("clf", classifier)])


@lru_cache(maxsize=4)
def load_model(path: str | None = None):
    model_path = MODEL_PATH if path is None else MODEL_PATH.__class__(path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}. From the repo root run: python -m src.train"
        )
    return joblib.load(model_path)


def _family(name: str) -> str:
    if name.startswith("heur__"):
        return "heuristic"
    if name.startswith("word__"):
        return "word"
    if name.startswith("char__"):
        return "char"
    return "other"


# Tiny display filter. The model still trains on these tokens. Hiding them in
# the explanation keeps "the" and "was" from crowding out the phrase that
# actually moved the score.
_DISPLAY_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "that", "the", "this", "to", "was", "were", "with",
    "you", "your",
}


def _worth_showing(feature: str) -> bool:
    parts = [part for part in feature.split() if part]
    if not parts:
        return False
    if len(parts) == 1:
        return parts[0] not in _DISPLAY_STOP
    return any(part not in _DISPLAY_STOP for part in parts)


def _pretty(name: str) -> str:
    for prefix in ("heur__", "word__tfidf__", "char__tfidf__", "word__", "char__"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.replace("_", " ").strip()


def _row_and_coef(pipeline: Pipeline, record: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    union = pipeline.named_steps["features"]
    classifier = pipeline.named_steps["clf"]
    names = [str(name) for name in union.get_feature_names_out()]
    transformed = union.transform([record])
    row = transformed.getrow(0) if sparse.issparse(transformed) else np.asarray(transformed[:1])
    values = row.toarray().ravel() if sparse.issparse(row) else np.asarray(row).ravel()
    coefficients = np.asarray(classifier.coef_).ravel()
    classes = list(classifier.classes_)
    # Binary logistic regression stores the weight vector for classes_[1].
    # Flip it if that column is somehow the legitimate class.
    if classes[1] != _PHISHING:
        coefficients = -coefficients
    if values.shape[0] != coefficients.shape[0]:
        raise RuntimeError(
            f"Feature width {values.shape[0]} does not match coefficient width {coefficients.shape[0]}."
        )
    return values, coefficients, names


def _cue(name: str, value: float, coefficient: float) -> dict:
    contribution = float(value * coefficient)
    return {
        "feature": _pretty(name),
        "family": _family(name),
        "value": round(float(value), 4),
        "coefficient": round(float(coefficient), 4),
        "contribution": round(contribution, 4),
    }


def _top_text_cues(
    names, values, coefficients, family: str, toward: str, limit: int, skip_stopwords: bool = False
) -> list[dict]:
    cues = []
    for name, value, coefficient in zip(names, values, coefficients):
        if _family(name) != family or value <= 0:
            continue
        contribution = float(value * coefficient)
        if toward == "phishing" and contribution <= 0:
            continue
        if toward == "legitimate" and contribution >= 0:
            continue
        cues.append(_cue(name, value, coefficient))
    cues.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    cues = [cue for cue in cues if abs(cue["contribution"]) >= 0.001]
    if skip_stopwords:
        filtered = [cue for cue in cues if _worth_showing(cue["feature"])]
        if filtered:
            cues = filtered
    return cues[:limit]


def _summary(label: str, phishing_probability: float, flags: list[dict], phishing_cues, legit_cues) -> str:
    legitimate_probability = 1.0 - phishing_probability
    if label == "phishing":
        sentence = f"Classified as phishing with a model probability of {phishing_probability:.0%}."
    else:
        sentence = (
            f"Classified as legitimate with a model probability of {legitimate_probability:.0%} "
            f"({phishing_probability:.0%} phishing)."
        )
    parts = [sentence]
    if max(phishing_probability, legitimate_probability) < 0.65:
        parts.append("The margin is thin, so a person should read the message before trusting the verdict.")
    if flags:
        titles = ", ".join(flag["title"].lower() for flag in flags[:3])
        parts.append(f"Security checks that fired: {titles}.")
    else:
        parts.append("None of the handcrafted security checks fired.")
    if label == "phishing" and phishing_cues:
        quoted = " and ".join(f"“{cue['feature']}”" for cue in phishing_cues[:2])
        parts.append(f"The strongest wording cues toward phishing include {quoted}.")
    elif label == "legitimate" and legit_cues:
        quoted = " and ".join(f"“{cue['feature']}”" for cue in legit_cues[:2])
        parts.append(f"The strongest wording cues toward legitimate include {quoted}.")
    return " ".join(parts)


def explain(pipeline: Pipeline, record: dict) -> dict:
    """Predict one email and return probabilities, flags, and n-gram contributions."""
    analysis: Analysis = analyze(record)
    classifier = pipeline.named_steps["clf"]
    probabilities = pipeline.predict_proba([record])[0]
    classes = list(classifier.classes_)
    phishing_probability = float(probabilities[classes.index(_PHISHING)])
    legitimate_probability = float(probabilities[classes.index(0)])
    predicted = int(pipeline.predict([record])[0])
    label = "phishing" if predicted == _PHISHING else "legitimate"

    values, coefficients, names = _row_and_coef(pipeline, record)
    phishing_cues = _top_text_cues(names, values, coefficients, "word", "phishing", 6, skip_stopwords=True)
    legitimate_cues = _top_text_cues(names, values, coefficients, "word", "legitimate", 6, skip_stopwords=True)
    phishing_chars = _top_text_cues(names, values, coefficients, "char", "phishing", 5)
    legitimate_chars = _top_text_cues(names, values, coefficients, "char", "legitimate", 5)

    heuristic = []
    for name, value, coefficient in zip(names, values, coefficients):
        if _family(name) != "heuristic" or value <= 0:
            continue
        heuristic.append(_cue(name, value, coefficient))
    heuristic.sort(key=lambda item: item["contribution"], reverse=True)

    return {
        "label": label,
        "phishing_probability": phishing_probability,
        "legitimate_probability": legitimate_probability,
        "confidence": phishing_probability if label == "phishing" else legitimate_probability,
        "low_margin": max(phishing_probability, legitimate_probability) < 0.65,
        "flags": analysis.flags,
        "heuristic_contributions": heuristic,
        "top_phishing_cues": phishing_cues,
        "top_legitimate_cues": legitimate_cues,
        "top_phishing_char_cues": phishing_chars,
        "top_legitimate_char_cues": legitimate_chars,
        "summary": _summary(label, phishing_probability, analysis.flags, phishing_cues, legitimate_cues),
    }
