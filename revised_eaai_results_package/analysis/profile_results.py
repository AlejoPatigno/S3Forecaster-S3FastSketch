from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "source_data"


def main() -> None:
    for path in sorted(SOURCE_DIR.glob("*.csv")):
        frame = pd.read_csv(path)
        print(f"\n=== {path.name} ===")
        print(f"shape={frame.shape} exact_duplicates={int(frame.duplicated().sum())}")
        for column in (
            "dataset",
            "model",
            "regime",
            "variant_key",
            "history_fraction",
            "rank",
        ):
            if column in frame:
                values = sorted(map(str, frame[column].dropna().unique()))
                print(f"{column} ({len(values)}): {values}")
        nulls = {
            column: int(count)
            for column, count in frame.isna().sum().items()
            if count
        }
        print(f"null_columns={nulls}")
        numeric = frame.select_dtypes(include=[np.number])
        nonfinite = {
            column: int((~np.isfinite(numeric[column].dropna())).sum())
            for column in numeric
            if (~np.isfinite(numeric[column].dropna())).any()
        }
        print(f"nonfinite={nonfinite}")
        mape_columns = [
            column
            for column in frame.columns
            if column in {"mape", "mape_median", "mape_mean", "mape_percent_median", "mape_percent_mean"}
        ]
        negative_mape = sum(int((frame[column] < 0).sum()) for column in mape_columns)
        print(f"negative_mape_rows={negative_mape}")


if __name__ == "__main__":
    main()
