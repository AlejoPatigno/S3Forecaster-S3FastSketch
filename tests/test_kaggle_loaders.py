from __future__ import annotations

import numpy as np

from s3paper.kaggle_loaders import load_m3_monthly


def test_load_m3_monthly_accepts_monash_tsf_layout(tmp_path):
    values_1 = ",".join(str(value) for value in range(1, 31))
    values_2 = ",".join(str(value) for value in range(101, 131))
    path = tmp_path / "m3_monthly_dataset.tsf"
    path.write_text(
        "\n".join(
            [
                "# Dataset Information",
                "@relation M3",
                "@attribute series_name string",
                "@attribute start_timestamp date",
                "@frequency monthly",
                "@horizon 18",
                "@missing false",
                "@equallength false",
                "@data",
                f"T1:1990-01-01 00-00-00:{values_1}",
                f"T2:1990-01-01 00-00-00:{values_2}",
            ]
        ),
        encoding="utf-8",
    )

    bundle = load_m3_monthly(
        tmp_path,
        minimum_train_length=1,
        maximum_train_length=None,
    )

    assert bundle.name == "M3_Monthly"
    assert bundle.series_ids == ["T1", "T2"]
    assert len(bundle.train_series_map["T1"]) == 12
    assert len(bundle.test_series_map["T1"]) == 18
    assert bundle.metadata["n_test"].tolist() == [18, 18]
    np.testing.assert_allclose(
        bundle.test_series_map["T1"].to_numpy(),
        np.arange(13, 31, dtype=float),
    )
