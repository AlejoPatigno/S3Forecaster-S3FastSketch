"""Build the source package directory used as a Kaggle input dataset."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "kaggle" / "source_dataset"
KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "alejopatio")
DATASET_ID = f"{KAGGLE_USERNAME}/s3forecaster-s3fastsketch-source"

FILES = [
    "README.md",
    "NOTEBOOK_AUDIT.md",
    "VALIDATION_REPORT.md",
    "KAGGLE.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-optional.txt",
]

DIRECTORIES = [
    "s3paper",
    "examples",
    "tests",
]


def copy_path(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main() -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for relative in FILES:
        copy_path(ROOT / relative, OUTPUT / relative)

    for relative in DIRECTORIES:
        copy_path(ROOT / relative, OUTPUT / relative)

    metadata = {
        "title": "S3Forecaster S3FastSketch Source",
        "id": DATASET_ID,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (OUTPUT / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote Kaggle source dataset package: {OUTPUT}")
    print(f"Dataset id: {DATASET_ID}")


if __name__ == "__main__":
    main()
