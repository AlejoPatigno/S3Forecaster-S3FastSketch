from __future__ import annotations

import numpy as np
import pandas as pd

from s3paper.kaggle_loaders import load_icmd_monthly, load_m3_monthly


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


def test_load_icmd_monthly_uses_product_prefix_series(tmp_path):
    months = pd.period_range("2021-01", periods=30, freq="M")
    rows = []

    for number, period in enumerate(months, start=1):
        rows.append(
            {
                "Periodo": int(period.strftime("%Y%m")),
                "Descripcion Producto": (
                    "LADRILLO FAROL PH DE 12 RAYADO ROJO PRIMERA ABC"
                ),
                "Valor Venta": float(number),
            }
        )
        rows.append(
            {
                "Periodo": int(period.strftime("%Y%m")),
                "Descripcion Producto": (
                    "LADRILLO FAROL PH DE 12 RAYADO ROJO COMERCIAL XYZ"
                ),
                "Valor Venta": float(number * 10),
            }
        )
        rows.append(
            {
                "Periodo": int(period.strftime("%Y%m")),
                "Descripcion Producto": "OTHER PRODUCT",
                "Valor Venta": 9999.0,
            }
        )

    csv_path = tmp_path / "df_preprocessed.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    bundle = load_icmd_monthly(
        tmp_path,
        test_horizon=6,
        print_summary=False,
    )

    series_id = "ICMD_LADRILLO_FAROL_PH12"
    expected = np.arange(1, 31, dtype=float) * 11.0

    assert bundle.name == "ICMD"
    assert bundle.series_ids == [series_id]
    assert len(bundle.train_series_map[series_id]) == 24
    assert len(bundle.test_series_map[series_id]) == 6
    assert bundle.metadata["test_horizon"].tolist() == [6]
    np.testing.assert_allclose(
        bundle.test_series_map[series_id].to_numpy(),
        expected[-6:],
    )
