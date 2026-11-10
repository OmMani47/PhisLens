"""Repository paths shared by the trainer, the CLI, and the demo app."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

DATA_PATH = DATA_DIR / "sample_emails.csv"
DEMOS_PATH = DATA_DIR / "demo_emails.json"
MODEL_PATH = MODEL_DIR / "pipeline.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
