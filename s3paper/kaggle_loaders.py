# ============================================================
# Archivo sugerido:
# s3paper/kaggle_loaders.py
# ============================================================

from __future__ import annotations

import fnmatch
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from pathlib import Path


_CIF_KEYWORDS = ("cif-dataset", "cif_dataset", "cif")
_SERIES_EXTENSIONS = {".tsf", ".ts"}


def _locate_cif_series_file(
    dataset_directory: str | Path,
) -> Path | None:
    """
    Locate the CIF TSF/TS file.

    First searches below dataset_directory. If that path does not exist or
    contains no compatible file, it searches recursively below /kaggle/input.
    """

    requested_root = Path(dataset_directory).expanduser()

    search_roots: list[Path] = []

    if requested_root.exists():
        search_roots.append(requested_root.resolve())

    kaggle_root = Path("/kaggle/input")
    if kaggle_root.exists():
        kaggle_root = kaggle_root.resolve()

        if kaggle_root not in search_roots:
            search_roots.append(kaggle_root)

    candidates: list[Path] = []

    for root in search_roots:
        candidates.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in _SERIES_EXTENSIONS
        )

    # Remove duplicated resolved paths.
    candidates = list(
        {
            candidate.resolve(): candidate.resolve()
            for candidate in candidates
        }.values()
    )

    if not candidates:
        return None

    def score(path: Path) -> tuple[int, int, int, int]:
        text = str(path).lower()

        keyword_score = sum(
            keyword in text
            for keyword in _CIF_KEYWORDS
        )

        monthly_score = int("monthly" in text)

        # Prefer TSF over TS because the CIF dataset normally uses TSF.
        tsf_score = int(path.suffix.lower() == ".ts")

        return (
            -keyword_score,
            -monthly_score,
            -tsf_score,
            len(text),
        )

    candidates.sort(key=score)

    selected = candidates[0]
    print(f"Selected CIF dataset file: {selected}")

    return selected

@dataclass(frozen=True)
class DatasetBundle:
    """
    Contenedor común utilizado por todos los notebooks.

    train_series_map:
        Series oficiales de entrenamiento o desarrollo.

    test_series_map:
        Bloques oficiales de prueba. Nunca deben participar en HPO.

    metadata:
        Información por serie: horizonte, fecha inicial, longitud, categoría, etc.
    """

    name: str
    train_series_map: dict[str, pd.Series]
    test_series_map: dict[str, pd.Series]
    metadata: pd.DataFrame
    seasonal_period: int = 12

    @property
    def series_ids(self) -> list[str]:
        return sorted(
            set(self.train_series_map).intersection(self.test_series_map)
        )

    def subset(self, series_ids: Sequence[str]) -> "DatasetBundle":
        selected = [str(series_id) for series_id in series_ids]
        selected_set = set(selected)

        train = {
            series_id: self.train_series_map[series_id]
            for series_id in selected
            if series_id in self.train_series_map
        }
        test = {
            series_id: self.test_series_map[series_id]
            for series_id in selected
            if series_id in self.test_series_map
        }

        metadata = self.metadata.copy()
        if "series_id" in metadata.columns:
            metadata = metadata[
                metadata["series_id"].astype(str).isin(selected_set)
            ].reset_index(drop=True)

        return DatasetBundle(
            name=self.name,
            train_series_map=train,
            test_series_map=test,
            metadata=metadata,
            seasonal_period=self.seasonal_period,
        )


# ============================================================
# Utilidades internas de archivos y nombres
# ============================================================

def _normalized_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _column_by_candidates(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    *,
    required: bool = False,
) -> str | None:
    normalized = {
        _normalized_name(column): str(column)
        for column in frame.columns
    }

    for candidate in candidates:
        key = _normalized_name(candidate)
        if key in normalized:
            return normalized[key]

    if required:
        raise KeyError(
            f"No se encontró ninguna columna entre {list(candidates)}. "
            f"Columnas disponibles: {list(frame.columns)}"
        )

    return None


def _find_file(
    root: str | Path,
    patterns: Sequence[str],
    *,
    required: bool = True,
) -> Path | None:
    root = Path(root)

    if root.is_file():
        return root

    if not root.exists():
        raise FileNotFoundError(f"No existe el directorio: {root}")

    files = sorted(path for path in root.rglob("*") if path.is_file())

    for pattern in patterns:
        pattern_lower = pattern.lower()

        for path in files:
            relative = str(path.relative_to(root)).lower()
            filename = path.name.lower()

            if (
                fnmatch.fnmatch(filename, pattern_lower)
                or fnmatch.fnmatch(relative, pattern_lower)
            ):
                return path

    if required:
        raise FileNotFoundError(
            f"No se encontró un archivo compatible con {patterns} dentro de {root}"
        )

    return None


def _read_csv_robust(path: Path) -> pd.DataFrame:
    errors: list[str] = []

    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return pd.read_csv(
                path,
                sep=None,
                engine="python",
                encoding=encoding,
            )
        except Exception as exc:
            errors.append(f"{encoding}: {exc}")

    raise RuntimeError(
        f"No fue posible leer {path}. Errores: {' | '.join(errors)}"
    )


def _read_table(
    path: str | Path,
    *,
    sheet_name: str | int | None = 0,
) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".csv", ".txt", ".tsv"}:
        return _read_csv_robust(path)

    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(path, sheet_name=sheet_name)

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)

    raise ValueError(f"Formato no soportado: {path}")


def _numeric_values(values: Iterable[Any]) -> np.ndarray:
    series = pd.Series(list(values))
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.dropna().to_numpy(dtype=float)


def _parse_locale_number(values: pd.Series) -> pd.Series:
    direct = pd.to_numeric(values, errors="coerce")

    if direct.notna().mean() >= 0.90:
        return direct

    text = values.astype(str).str.strip()
    text = text.str.replace(r"[$\s]", "", regex=True)

    both = text.str.contains(",", regex=False) & text.str.contains(".", regex=False)

    colombian = text.copy()
    colombian.loc[both] = (
        colombian.loc[both]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )

    only_comma = colombian.str.contains(",", regex=False) & ~colombian.str.contains(
        ".", regex=False
    )
    colombian.loc[only_comma] = colombian.loc[only_comma].str.replace(
        ",", ".", regex=False
    )

    return pd.to_numeric(colombian, errors="coerce")


def _monthly_index(
    start: Any,
    length: int,
) -> pd.DatetimeIndex:
    try:
        first_period = pd.Period(pd.Timestamp(start), freq="M")
    except Exception:
        first_period = pd.Period("2000-01", freq="M")

    return pd.period_range(
        start=first_period,
        periods=int(length),
        freq="M",
    ).to_timestamp("M")


def _make_monthly_series(
    values: Iterable[Any],
    *,
    start: Any = "2000-01",
    name: str | None = None,
) -> pd.Series:
    numeric = _numeric_values(values)

    if len(numeric) == 0:
        raise ValueError(f"La serie {name!r} no contiene valores numéricos.")

    return pd.Series(
        numeric,
        index=_monthly_index(start, len(numeric)),
        name=name,
        dtype=float,
    )


def _infer_id_column(frame: pd.DataFrame) -> str:
    candidate = _column_by_candidates(
        frame,
        (
            "series_id",
            "series",
            "series_name",
            "id",
            "m4id",
            "m3id",
            "v1",
            "name",
        ),
    )
    return candidate or str(frame.columns[0])


def _value_columns(frame: pd.DataFrame, id_column: str) -> list[str]:
    numeric_named = [
        str(column)
        for column in frame.columns
        if re.fullmatch(r"\d+", str(column).strip())
    ]

    if numeric_named:
        return sorted(numeric_named, key=lambda value: int(value))

    excluded = {
        _normalized_name(value)
        for value in (
            id_column,
            "series_id",
            "series",
            "series_name",
            "id",
            "frequency",
            "category",
            "type",
            "n",
            "nf",
            "horizon",
            "forecast_horizon",
            "starting_year",
            "starting_month",
            "starting_period",
            "start_timestamp",
            "starting_date",
        )
    }

    selected: list[str] = []

    for column in frame.columns:
        if _normalized_name(column) in excluded:
            continue

        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().any():
            selected.append(str(column))

    return selected


def _extract_start_date(
    row: pd.Series | None,
    *,
    default_start: str = "2000-01",
) -> pd.Timestamp:
    if row is None:
        return pd.Timestamp(default_start)

    normalized = {
        _normalized_name(column): column
        for column in row.index
    }

    for candidate in (
        "startingdate",
        "startdate",
        "starttimestamp",
        "start",
    ):
        if candidate in normalized:
            value = row[normalized[candidate]]
            try:
                return pd.Timestamp(value)
            except Exception:
                pass

    year_column = next(
        (
            normalized[key]
            for key in ("startingyear", "startyear", "year")
            if key in normalized
        ),
        None,
    )
    month_column = next(
        (
            normalized[key]
            for key in ("startingmonth", "startmonth", "month", "startingperiod")
            if key in normalized
        ),
        None,
    )

    if year_column is not None:
        year = int(float(row[year_column]))
        month = (
            int(float(row[month_column]))
            if month_column is not None and pd.notna(row[month_column])
            else 1
        )
        return pd.Timestamp(year=year, month=max(1, min(month, 12)), day=1)

    return pd.Timestamp(default_start)


def _validate_bundle(
    bundle: DatasetBundle,
    *,
    minimum_train_length: int = 24,
    maximum_train_length: int | None = 199,
    maximum_series: int | None = None,
    selected_series: Sequence[str] | None = None,
) -> DatasetBundle:
    selected_set = (
        {str(value) for value in selected_series}
        if selected_series is not None
        else None
    )

    train_map: dict[str, pd.Series] = {}
    test_map: dict[str, pd.Series] = {}
    metadata_rows: list[dict[str, Any]] = []

    for series_id in bundle.series_ids:
        if selected_set is not None and series_id not in selected_set:
            continue

        train = bundle.train_series_map[series_id].astype(float).sort_index()
        test = bundle.test_series_map[series_id].astype(float).sort_index()

        if not np.isfinite(train.to_numpy()).all():
            raise ValueError(f"{bundle.name}/{series_id}: train contiene NaN o infinitos.")

        if not np.isfinite(test.to_numpy()).all():
            raise ValueError(f"{bundle.name}/{series_id}: test contiene NaN o infinitos.")

        if len(train) < int(minimum_train_length):
            continue

        if maximum_train_length is not None and len(train) > maximum_train_length:
            train = train.iloc[-int(maximum_train_length):].copy()

        if len(test) == 0:
            continue

        if train.index.max() >= test.index.min():
            test = pd.Series(
                test.to_numpy(dtype=float),
                index=_monthly_index(
                    train.index[-1] + pd.offsets.MonthEnd(1),
                    len(test),
                ),
                name=test.name,
            )

        train_map[series_id] = train
        test_map[series_id] = test

        metadata_rows.append(
            {
                "dataset": bundle.name,
                "series_id": series_id,
                "n_train": len(train),
                "n_test": len(test),
                "train_start": train.index.min(),
                "train_end": train.index.max(),
                "test_start": test.index.min(),
                "test_end": test.index.max(),
                "seasonal_period": bundle.seasonal_period,
            }
        )

        if maximum_series is not None and len(train_map) >= int(maximum_series):
            break

    if not train_map:
        raise ValueError(
            f"Ninguna serie elegible fue cargada para {bundle.name}."
        )

    return DatasetBundle(
        name=bundle.name,
        train_series_map=train_map,
        test_series_map=test_map,
        metadata=pd.DataFrame(metadata_rows),
        seasonal_period=bundle.seasonal_period,
    )


# ============================================================
# Formato ancho train/test: M3, M4 y variantes de Tourism/CIF
# ============================================================

def _load_wide_train_test(
    train_path: str | Path,
    test_path: str | Path,
    *,
    dataset_name: str,
    info_path: str | Path | None = None,
    seasonal_period: int = 12,
    default_start: str = "2000-01",
) -> DatasetBundle:
    train_frame = _read_table(train_path)
    test_frame = _read_table(test_path)

    train_id_column = _infer_id_column(train_frame)
    test_id_column = _infer_id_column(test_frame)

    info_frame = (
        _read_table(info_path)
        if info_path is not None and Path(info_path).exists()
        else pd.DataFrame()
    )
    info_id_column = _infer_id_column(info_frame) if not info_frame.empty else None

    test_lookup = {
        str(row[test_id_column]): row
        for _, row in test_frame.iterrows()
    }

    info_lookup = (
        {
            str(row[info_id_column]): row
            for _, row in info_frame.iterrows()
        }
        if info_id_column is not None
        else {}
    )

    train_columns = _value_columns(train_frame, train_id_column)
    test_columns = _value_columns(test_frame, test_id_column)

    train_map: dict[str, pd.Series] = {}
    test_map: dict[str, pd.Series] = {}
    metadata_rows: list[dict[str, Any]] = []

    for _, train_row in train_frame.iterrows():
        series_id = str(train_row[train_id_column])

        if series_id not in test_lookup:
            continue

        test_row = test_lookup[series_id]
        info_row = info_lookup.get(series_id)

        train_values = _numeric_values(train_row[train_columns])
        test_values = _numeric_values(test_row[test_columns])

        start = _extract_start_date(
            info_row,
            default_start=default_start,
        )

        train_series = _make_monthly_series(
            train_values,
            start=start,
            name=series_id,
        )
        test_series = pd.Series(
            test_values,
            index=_monthly_index(
                train_series.index[-1] + pd.offsets.MonthEnd(1),
                len(test_values),
            ),
            name=series_id,
            dtype=float,
        )

        train_map[series_id] = train_series
        test_map[series_id] = test_series

        metadata_rows.append(
            {
                "dataset": dataset_name,
                "series_id": series_id,
                "n_train_original": len(train_values),
                "forecast_horizon": len(test_values),
                "start_timestamp": start,
            }
        )

    return DatasetBundle(
        name=dataset_name,
        train_series_map=train_map,
        test_series_map=test_map,
        metadata=pd.DataFrame(metadata_rows),
        seasonal_period=seasonal_period,
    )


# ============================================================
# Parser TSF para Tourism y CIF Monash
# ============================================================




def _read_tsf_lines(
    path: str | Path,
    *,
    encodings: tuple[str, ...] = (
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin-1",
    ),
) -> tuple[list[str], str]:
    """
    Read a TSF file using a deterministic encoding fallback.

    Returns
    -------
    lines:
        Decoded text split into lines.

    encoding:
        Encoding that successfully decoded the file.
    """

    path = Path(path)
    errors: list[str] = []

    for encoding in encodings:
        try:
            text = path.read_text(encoding=encoding)
            return text.splitlines(), encoding
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")

    raise UnicodeError(
        f"Could not decode TSF file {path} with encodings "
        f"{encodings}. Errors: {' | '.join(errors)}"
    )


def _parse_tsf_date(value: Any) -> pd.Timestamp:
    """
    Parse TSF date attributes.

    Some Monash TSF files use values such as:

        1979-01-01 00-00-00

    instead of a conventional ISO time representation.
    """

    text = str(value).strip()

    # Remove nonstandard zero-time suffixes.
    text = re.sub(
        r"\s+00[-:]00[-:]00$",
        "",
        text,
    )

    parsed = pd.to_datetime(text, errors="coerce")

    if pd.notna(parsed):
        return pd.Timestamp(parsed)

    match = re.match(
        r"^\s*(\d{4})-(\d{1,2})-(\d{1,2})",
        text,
    )

    if match is None:
        raise ValueError(
            f"Could not parse TSF date attribute: {value!r}"
        )

    year, month, day = map(int, match.groups())

    return pd.Timestamp(
        year=year,
        month=month,
        day=day,
    )


def read_tsf(
    path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Parse a Monash-style TSF file.

    This function is compatible with `_bundle_from_tsf()` because it returns
    a DataFrame containing a `series_value` column.

    Parameters
    ----------
    path:
        Path to the TSF file.

    Returns
    -------
    frame:
        DataFrame with TSF attributes and one NumPy array per row in
        `series_value`.

    metadata:
        Global TSF metadata, including the detected encoding.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    lines, used_encoding = _read_tsf_lines(path)

    metadata: dict[str, Any] = {
        "relation": None,
        "frequency": None,
        "horizon": None,
        "missing": None,
        "equallength": None,
        "attributes": [],
        "encoding": used_encoding,
        "source_file": str(path),
    }

    attributes: list[tuple[str, str]] = []
    rows: list[dict[str, Any]] = []
    in_data = False

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        lower = line.lower()

        if not in_data:
            if lower.startswith("@relation"):
                parts = line.split(maxsplit=1)
                metadata["relation"] = (
                    parts[1].strip()
                    if len(parts) > 1
                    else None
                )

            elif lower.startswith("@attribute"):
                parts = line.split(maxsplit=2)

                if len(parts) != 3:
                    raise ValueError(
                        f"Invalid @attribute declaration at line "
                        f"{line_number}: {line!r}"
                    )

                attribute_name = parts[1].strip()
                attribute_type = parts[2].strip().lower()

                attributes.append(
                    (attribute_name, attribute_type)
                )
                metadata["attributes"].append(
                    (attribute_name, attribute_type)
                )

            elif lower.startswith("@frequency"):
                parts = line.split(maxsplit=1)
                metadata["frequency"] = (
                    parts[1].strip().lower()
                    if len(parts) > 1
                    else None
                )

            elif lower.startswith("@horizon"):
                parts = line.split(maxsplit=1)

                if len(parts) > 1:
                    metadata["horizon"] = int(
                        float(parts[1].strip())
                    )

            elif lower.startswith("@missing"):
                parts = line.split(maxsplit=1)
                metadata["missing"] = (
                    len(parts) > 1
                    and parts[1].strip().lower() == "true"
                )

            elif lower.startswith("@equallength"):
                parts = line.split(maxsplit=1)
                metadata["equallength"] = (
                    len(parts) > 1
                    and parts[1].strip().lower() == "true"
                )

            elif lower == "@data":
                in_data = True

            continue

        number_of_attributes = len(attributes)

        # The last field contains the comma-separated observations.
        # maxsplit avoids splitting additional colon characters unnecessarily.
        parts = line.split(":", number_of_attributes)

        if len(parts) != number_of_attributes + 1:
            raise ValueError(
                f"Invalid TSF row at line {line_number}: expected "
                f"{number_of_attributes + 1} fields, received "
                f"{len(parts)}. Row={line[:180]!r}"
            )

        row: dict[str, Any] = {}

        for (
            attribute_name,
            attribute_type,
        ), raw_value in zip(
            attributes,
            parts[:number_of_attributes],
        ):
            value = raw_value.strip()

            if value in {"", "?"}:
                row[attribute_name] = np.nan

            elif attribute_type in {
                "numeric",
                "integer",
                "int",
                "real",
                "float",
                "double",
            }:
                row[attribute_name] = float(value)

            elif attribute_type in {
                "date",
                "datetime",
                "timestamp",
            }:
                row[attribute_name] = _parse_tsf_date(value)

            else:
                row[attribute_name] = value

        observations: list[float] = []

        for raw_value in parts[-1].split(","):
            value = raw_value.strip()

            if value in {"", "?"}:
                observations.append(np.nan)
            else:
                try:
                    observations.append(float(value))
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid numeric value at line "
                        f"{line_number}: {value!r}"
                    ) from exc

        if not observations:
            raise ValueError(
                f"No observations found at TSF line {line_number}."
            )

        row["series_value"] = np.asarray(
            observations,
            dtype=float,
        )
        rows.append(row)

    if not in_data:
        raise ValueError(
            f"The TSF file {path} does not contain an @data declaration."
        )

    if not rows:
        raise ValueError(
            f"No time series were parsed from {path}."
        )

    frame = pd.DataFrame(rows)

    print("Selected TSF file:", path)
    print("TSF encoding:", used_encoding)
    print("Parsed series:", len(frame))
    print("Global horizon:", metadata.get("horizon"))
    print("Frequency:", metadata.get("frequency"))

    return frame, metadata

def _bundle_from_tsf(
    path: str | Path,
    *,
    dataset_name: str,
    default_horizon: int,
    seasonal_period: int = 12,
    missing_policy: str = "raise",
) -> DatasetBundle:
    frame, file_metadata = read_tsf(path)

    id_column = _column_by_candidates(
        frame,
        ("series_name", "series_id", "series", "id"),
    )
    start_column = _column_by_candidates(
        frame,
        ("start_timestamp", "starting_date", "start"),
    )
    horizon_column = _column_by_candidates(
        frame,
        ("forecast_horizon", "horizon"),
    )

    train_map: dict[str, pd.Series] = {}
    test_map: dict[str, pd.Series] = {}
    metadata_rows: list[dict[str, Any]] = []

    for row_number, row in frame.iterrows():
        series_id = (
            str(row[id_column])
            if id_column is not None
            else f"{dataset_name}_{row_number:05d}"
        )

        values = np.asarray(
            row["series_value"],
            dtype=float,
        )

        missing_count = int(
            np.isnan(values).sum()
        )

        if missing_count > 0:
            if missing_policy == "raise":
                raise ValueError(
                    f"{dataset_name}/{series_id} contains "
                    f"{missing_count} missing values."
                )

            if missing_policy == "interpolate":
                values = (
                    pd.Series(values, dtype=float)
                    .interpolate(
                        method="linear",
                        limit_direction="both",
                    )
                    .ffill()
                    .bfill()
                    .to_numpy(dtype=float)
                )

            elif missing_policy == "drop":
                values = values[
                    np.isfinite(values)
                ]

            else:
                raise ValueError(
                    f"Invalid missing_policy={missing_policy!r}. "
                    "Expected 'raise', 'interpolate', or 'drop'."
                )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{dataset_name}/{series_id} still contains "
                "non-finite values after preprocessing."
            )

        horizon = (
            int(row[horizon_column])
            if horizon_column is not None and pd.notna(row[horizon_column])
            else int(file_metadata.get("horizon", default_horizon))
        )

        if horizon <= 0 or len(values) <= horizon:
            continue

        start = (
            row[start_column]
            if start_column is not None and pd.notna(row[start_column])
            else "2000-01"
        )

        full_series = _make_monthly_series(
            values,
            start=start,
            name=series_id,
        )

        train_map[series_id] = full_series.iloc[:-horizon].copy()
        test_map[series_id] = full_series.iloc[-horizon:].copy()

        metadata_rows.append(
            {
                "dataset": dataset_name,
                "series_id": series_id,
                "forecast_horizon": horizon,
                "start_timestamp": full_series.index[0],
                "frequency": file_metadata.get(
                    "frequency",
                    "monthly",
                ),
                "missing_values_imputed": missing_count,
                "source_file": file_metadata.get(
                    "source_file",
                    str(path),
                ),
                "source_encoding": file_metadata.get(
                    "encoding",
                ),
            }
        )

    return DatasetBundle(
        name=dataset_name,
        train_series_map=train_map,
        test_series_map=test_map,
        metadata=pd.DataFrame(metadata_rows),
        seasonal_period=seasonal_period,
    )


# ============================================================
# Formato genérico: long, series_value o ancho completo
# ============================================================

def _bundle_from_generic_table(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    default_horizon: int,
    seasonal_period: int = 12,
) -> DatasetBundle:
    id_column = _column_by_candidates(
        frame,
        ("series_id", "series_name", "series", "id", "name"),
    )
    date_column = _column_by_candidates(
        frame,
        ("date", "timestamp", "period", "month", "fecha"),
    )
    value_column = _column_by_candidates(
        frame,
        ("value", "target", "y", "series_value_numeric"),
    )
    series_value_column = _column_by_candidates(
        frame,
        ("series_value", "values", "data"),
    )
    horizon_column = _column_by_candidates(
        frame,
        ("forecast_horizon", "horizon", "nf", "h"),
    )
    start_column = _column_by_candidates(
        frame,
        ("start_timestamp", "starting_date", "start"),
    )

    train_map: dict[str, pd.Series] = {}
    test_map: dict[str, pd.Series] = {}
    metadata_rows: list[dict[str, Any]] = []

    # --------------------------------------------------------
    # Formato largo: series_id, date, value
    # --------------------------------------------------------
    if id_column and date_column and value_column:
        frame = frame.copy()
        frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
        frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
        frame = frame.dropna(subset=[date_column, value_column])

        for series_id, group in frame.groupby(id_column):
            group = group.sort_values(date_column)
            horizon = (
                int(group[horizon_column].iloc[0])
                if horizon_column is not None
                else int(default_horizon)
            )

            values = group[value_column].to_numpy(dtype=float)
            if len(values) <= horizon:
                continue

            full = pd.Series(
                values,
                index=pd.DatetimeIndex(group[date_column]),
                name=str(series_id),
            )
            full.index = full.index.to_period("M").to_timestamp("M")
            full = full.groupby(level=0).sum().sort_index()

            train_map[str(series_id)] = full.iloc[:-horizon].copy()
            test_map[str(series_id)] = full.iloc[-horizon:].copy()

            metadata_rows.append(
                {
                    "dataset": dataset_name,
                    "series_id": str(series_id),
                    "forecast_horizon": horizon,
                }
            )

    # --------------------------------------------------------
    # Formato Monash convertido: series_value como texto/lista
    # --------------------------------------------------------
    elif series_value_column:
        for row_number, row in frame.iterrows():
            series_id = (
                str(row[id_column])
                if id_column is not None
                else f"{dataset_name}_{row_number:05d}"
            )

            raw_values = row[series_value_column]

            if isinstance(raw_values, str):
                values = [
                    np.nan if token.strip() == "?" else float(token)
                    for token in re.split(r"[,;\s]+", raw_values.strip())
                    if token.strip()
                ]
            else:
                values = list(raw_values)

            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]

            horizon = (
                int(row[horizon_column])
                if horizon_column is not None and pd.notna(row[horizon_column])
                else int(default_horizon)
            )

            if len(values) <= horizon:
                continue

            start = (
                row[start_column]
                if start_column is not None and pd.notna(row[start_column])
                else "2000-01"
            )

            full = _make_monthly_series(
                values,
                start=start,
                name=series_id,
            )

            train_map[series_id] = full.iloc[:-horizon].copy()
            test_map[series_id] = full.iloc[-horizon:].copy()

            metadata_rows.append(
                {
                    "dataset": dataset_name,
                    "series_id": series_id,
                    "forecast_horizon": horizon,
                }
            )

    # --------------------------------------------------------
    # Formato ancho: una fila por serie y todas las observaciones
    # --------------------------------------------------------
    else:
        id_column = id_column or _infer_id_column(frame)
        value_columns = _value_columns(frame, id_column)

        for row_number, row in frame.iterrows():
            series_id = str(row[id_column])
            values = _numeric_values(row[value_columns])

            horizon = (
                int(row[horizon_column])
                if horizon_column is not None and pd.notna(row[horizon_column])
                else int(default_horizon)
            )

            if len(values) <= horizon:
                continue

            start = _extract_start_date(
                row,
                default_start="2000-01",
            )
            full = _make_monthly_series(
                values,
                start=start,
                name=series_id,
            )

            train_map[series_id] = full.iloc[:-horizon].copy()
            test_map[series_id] = full.iloc[-horizon:].copy()

            metadata_rows.append(
                {
                    "dataset": dataset_name,
                    "series_id": series_id,
                    "forecast_horizon": horizon,
                }
            )

    return DatasetBundle(
        name=dataset_name,
        train_series_map=train_map,
        test_series_map=test_map,
        metadata=pd.DataFrame(metadata_rows),
        seasonal_period=seasonal_period,
    )


# ============================================================
# Loader M4 Monthly
# ============================================================

def load_m4_monthly(
    dataset_directory: str | Path,
    *,
    minimum_train_length: int = 36,
    maximum_train_length: int | None = 199,
    maximum_series: int | None = None,
    selected_series: Sequence[str] | None = None,
) -> DatasetBundle:
    root = Path(dataset_directory)

    train_path = _find_file(
        root,
        (
            "monthly-train.csv",
            "m-train.csv",
            "*monthly*train*.csv",
        ),
    )
    test_path = _find_file(
        root,
        (
            "monthly-test.csv",
            "m-test.csv",
            "*monthly*test*.csv",
        ),
    )
    info_path = _find_file(
        root,
        (
            "m4-info.csv",
            "*info*.csv",
        ),
        required=False,
    )

    bundle = _load_wide_train_test(
        train_path,
        test_path,
        dataset_name="M4_Monthly",
        info_path=info_path,
        seasonal_period=12,
        default_start="2000-01",
    )

    return _validate_bundle(
        bundle,
        minimum_train_length=minimum_train_length,
        maximum_train_length=maximum_train_length,
        maximum_series=maximum_series,
        selected_series=selected_series,
    )


# ============================================================
# Loader M3 Monthly
# ============================================================

def load_m3_monthly(
    dataset_directory: str | Path,
    *,
    minimum_train_length: int = 36,
    maximum_train_length: int | None = 199,
    maximum_series: int | None = None,
    selected_series: Sequence[str] | None = None,
    default_horizon: int = 18,
) -> DatasetBundle:
    root = Path(dataset_directory)

    train_path = _find_file(
        root,
        (
            "*month*train*.csv",
            "*monthly*train*.csv",
            "m3monthtrain.csv",
        ),
        required=False,
    )
    test_path = _find_file(
        root,
        (
            "*month*test*.csv",
            "*monthly*test*.csv",
            "m3monthtest.csv",
        ),
        required=False,
    )

    if train_path is not None and test_path is not None:
        info_path = _find_file(
            root,
            ("*info*.csv", "*metadata*.csv"),
            required=False,
        )

        bundle = _load_wide_train_test(
            train_path,
            test_path,
            dataset_name="M3_Monthly",
            info_path=info_path,
            seasonal_period=12,
        )

    else:
        tsf_path = _find_file(
            root,
            (
                "m3_monthly_dataset.tsf",
                "*m3*monthly*.tsf",
                "*monthly*.tsf",
                "*.tsf",
            ),
            required=False,
        )

        if tsf_path is not None:
            bundle = _bundle_from_tsf(
                tsf_path,
                dataset_name="M3_Monthly",
                default_horizon=default_horizon,
                seasonal_period=12,
            )
        else:
            workbook = _find_file(
                root,
                (
                    "m3c.xls",
                    "m3c.xlsx",
                    "*m3*.xls",
                    "*m3*.xlsx",
                    "*month*.csv",
                ),
            )

            if workbook.suffix.lower() in {".xls", ".xlsx", ".xlsm"}:
                excel = pd.ExcelFile(workbook)
                sheet_name = next(
                    (
                        sheet
                        for sheet in excel.sheet_names
                        if "month" in sheet.lower()
                    ),
                    excel.sheet_names[0],
                )
                frame = pd.read_excel(workbook, sheet_name=sheet_name)
            else:
                frame = _read_table(workbook)

            bundle = _bundle_from_generic_table(
                frame,
                dataset_name="M3_Monthly",
                default_horizon=default_horizon,
                seasonal_period=12,
            )

    return _validate_bundle(
        bundle,
        minimum_train_length=minimum_train_length,
        maximum_train_length=maximum_train_length,
        maximum_series=maximum_series,
        selected_series=selected_series,
    )


# ============================================================
# Loader Tourism Monthly
# ============================================================

def load_tourism_monthly(
    dataset_directory: str | Path,
    *,
    minimum_train_length: int = 36,
    maximum_train_length: int | None = 199,
    maximum_series: int | None = None,
    selected_series: Sequence[str] | None = None,
    default_horizon: int = 24,
) -> DatasetBundle:
    root = Path(dataset_directory)

    train_path = _find_file(
        root,
        ("*monthly*train*.csv", "*tourism*train*.csv"),
        required=False,
    )
    test_path = _find_file(
        root,
        ("*monthly*test*.csv", "*tourism*test*.csv"),
        required=False,
    )

    if train_path is not None and test_path is not None:
        bundle = _load_wide_train_test(
            train_path,
            test_path,
            dataset_name="Tourism_Monthly",
            seasonal_period=12,
        )
    else:
        tsf_path = _find_file(
            root,
            ("*tourism*monthly*.tsf", "*.tsf"),
            required=False,
        )

        if tsf_path is not None:
            bundle = _bundle_from_tsf(
            tsf_path,
            dataset_name="Tourism_Monthly",
            default_horizon=default_horizon,
            seasonal_period=12,
            missing_policy="interpolate",
            )
        else:
            table_path = _find_file(
                root,
                (
                    "*tourism*monthly*.csv",
                    "*tourism*.csv",
                    "*.xlsx",
                    "*.csv",
                ),
            )
            bundle = _bundle_from_generic_table(
                _read_table(table_path),
                dataset_name="Tourism_Monthly",
                default_horizon=default_horizon,
                seasonal_period=12,
            )

    return _validate_bundle(
        bundle,
        minimum_train_length=minimum_train_length,
        maximum_train_length=maximum_train_length,
        maximum_series=maximum_series,
        selected_series=selected_series,
    )


# ============================================================
# Loader CIF 2016
# ============================================================

def load_cif_2016(
    dataset_directory,
    minimum_train_length=36,
    maximum_train_length=199,
    maximum_series=None,
    selected_series=None,
    default_horizon=12,
):
    root = Path(dataset_directory).expanduser()

    # ========================================================
    # CIF is normally distributed as a TSF file
    # ========================================================

    series_path = _locate_cif_series_file(root)

    if series_path is not None:
        bundle = _bundle_from_tsf(
            series_path,
            dataset_name="CIF 2016",
            default_horizon=default_horizon,
            seasonal_period=12,
            missing_policy="interpolate",
            )
        return _validate_bundle(
        bundle,
        minimum_train_length=minimum_train_length,
        maximum_train_length=maximum_train_length,
        maximum_series=maximum_series,
        selected_series=selected_series,
    )

    # ========================================================
    # Existing CSV/XLSX fallback
    # ========================================================

    train_path = _find_file(
        root,
        (
            "*train*.csv",
            "*train*.xlsx",
            "*training*.csv",
            "*training*.xlsx",
        ),
        required=False,
    )

    test_path = _find_file(
        root,
        (
            "*test*.csv",
            "*test*.xlsx",
            "*evaluation*.csv",
            "*evaluation*.xlsx",
        ),
        required=False,
    )

    if train_path is not None and test_path is not None:
        return _load_wide_train_test(
            train_path=train_path,
            test_path=test_path,
            dataset_name="CIF 2016",
            minimum_train_length=minimum_train_length,
            maximum_train_length=maximum_train_length,
            maximum_series=maximum_series,
            selected_series=selected_series,
        )

    table_path = _find_file(
        root,
        (
            "*cif*.xlsx",
            "*cif*.csv",
            "*.xlsx",
            "*.csv",
        ),
        required=True,
    )

    return _bundle_from_generic_table(
        table_path,
        dataset_name="CIF 2016",
        minimum_train_length=minimum_train_length,
        maximum_train_length=maximum_train_length,
        maximum_series=maximum_series,
        selected_series=selected_series,
        default_horizon=default_horizon,
    )


# ============================================================
# Loader ICMD / CASAGRES
# ============================================================

def _parse_month_column(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()

    yyyymm = text.str.fullmatch(r"\d{6}")

    parsed = pd.to_datetime(text, errors="coerce")

    if yyyymm.any():
        parsed.loc[yyyymm] = pd.to_datetime(
            text.loc[yyyymm],
            format="%Y%m",
            errors="coerce",
        )

    return parsed.dt.to_period("M").dt.to_timestamp("M")


def load_icmd_monthly(
    dataset_path: str | Path,
    *,
    date_column: str | None = None,
    target_column: str | None = None,
    series_id_column: str | None = None,
    selected_series: Sequence[str] | None = None,
    test_horizon: int = 10,
    minimum_train_length: int = 24,
    maximum_train_length: int | None = 199,
    maximum_series: int | None = None,
    positive_sales_only: bool = True,
    fill_missing_months: float | None = 0.0,
    sheet_name: str | int = 0,
) -> DatasetBundle:
    path = Path(dataset_path)

    if path.is_dir():
        path = _find_file(
            path,
            (
                "*factura*.parquet",
                "*factura*.xlsx",
                "*factura*.xls",
                "*factura*.csv",
                "*casagres*.xlsx",
                "*icmd*.xlsx",
                "*.parquet",
                "*.xlsx",
                "*.xls",
                "*.csv",
            ),
        )

    frame = _read_table(path, sheet_name=sheet_name)

    date_column = date_column or _column_by_candidates(
        frame,
        (
            "Periodo",
            "Fecha",
            "Date",
            "Mes",
            "AñoMes",
            "YearMonth",
        ),
        required=True,
    )
    target_column = target_column or _column_by_candidates(
        frame,
        (
            "Valor Venta",
            "ValorVenta",
            "valor_venta",
            "SalesValue",
            "Valor Neto",
            "Venta Neta",
        ),
        required=True,
    )
    series_id_column = series_id_column or _column_by_candidates(
        frame,
        (
            "Material",
            "Referencia",
            "Producto",
            "Codigo Material",
            "Código Material",
            "Product",
            "ProductID",
        ),
        required=False,
    )

    working = frame.copy()
    working["_month"] = _parse_month_column(working[date_column])
    working["_target"] = _parse_locale_number(working[target_column])
    working = working.dropna(subset=["_month", "_target"])

    if positive_sales_only:
        working = working[working["_target"] > 0.0]

    if series_id_column is None:
        working["_series_id"] = "ICMD"
        series_id_column = "_series_id"
    else:
        working[series_id_column] = working[series_id_column].astype(str)

    if selected_series is not None:
        allowed = {str(value) for value in selected_series}
        working = working[working[series_id_column].isin(allowed)]

    monthly = (
        working.groupby([series_id_column, "_month"], as_index=False)["_target"]
        .sum()
        .sort_values([series_id_column, "_month"])
    )

    train_map: dict[str, pd.Series] = {}
    test_map: dict[str, pd.Series] = {}
    metadata_rows: list[dict[str, Any]] = []

    for series_id, group in monthly.groupby(series_id_column):
        group = group.sort_values("_month")

        start = pd.Period(group["_month"].min(), freq="M")
        end = pd.Period(group["_month"].max(), freq="M")
        complete_index = pd.period_range(start=start, end=end, freq="M").to_timestamp("M")

        series = pd.Series(
            group["_target"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(group["_month"]),
            name=str(series_id),
        )
        series = series.groupby(level=0).sum().reindex(complete_index)

        if fill_missing_months is None and series.isna().any():
            raise ValueError(
                f"ICMD/{series_id} contiene meses faltantes. "
                "Defina fill_missing_months explícitamente."
            )

        if fill_missing_months is not None:
            series = series.fillna(float(fill_missing_months))

        if len(series) <= int(test_horizon):
            continue

        train_map[str(series_id)] = series.iloc[:-int(test_horizon)].copy()
        test_map[str(series_id)] = series.iloc[-int(test_horizon):].copy()

        metadata_rows.append(
            {
                "dataset": "ICMD",
                "series_id": str(series_id),
                "n_transactions": int(
                    (working[series_id_column] == str(series_id)).sum()
                ),
                "n_months": len(series),
                "forecast_horizon": int(test_horizon),
                "start_timestamp": series.index.min(),
                "end_timestamp": series.index.max(),
            }
        )

def load_cif_2016(
    root: str | Path,
    *,
    minimum_train_length: int = 24,
    maximum_train_length: int | None = 199,
    maximum_series: int | None = None,
    selected_series: Sequence[str] | None = None,
    default_horizon: int = 6,
) -> DatasetBundle:
    table_path = _find_file(
        root,
        (
            "*cif*.xlsx",
            "*cif*.csv",
            "*.xlsx",
            "*.csv",
        ),
        required=True,
    )

    return _bundle_from_generic_table(
        table_path,
        dataset_name="CIF 2016",
        minimum_train_length=minimum_train_length,
        maximum_train_length=maximum_train_length,
        maximum_series=maximum_series,
        selected_series=selected_series,
        default_horizon=default_horizon,
    )


# ============================================================
# Loader ICMD / CASAGRES
# ============================================================

def _parse_month_column(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()

    yyyymm = text.str.fullmatch(r"\d{6}")

    parsed = pd.to_datetime(text, errors="coerce")

    if yyyymm.any():
        parsed.loc[yyyymm] = pd.to_datetime(
            text.loc[yyyymm],
            format="%Y%m",
            errors="coerce",
        )

    return parsed.dt.to_period("M").dt.to_timestamp("M")


def load_icmd_monthly(
    dataset_path: str | Path,
    *,
    date_column: str | None = None,
    target_column: str | None = None,
    series_id_column: str | None = None,
    selected_series: Sequence[str] | None = None,
    test_horizon: int = 10,
    minimum_train_length: int = 24,
    maximum_train_length: int | None = 199,
    maximum_series: int | None = None,
    positive_sales_only: bool = True,
    fill_missing_months: float | None = 0.0,
    sheet_name: str | int = 0,
) -> DatasetBundle:
    path = Path(dataset_path)

    if path.is_dir():
        path = _find_file(
            path,
            (
                "*factura*.parquet",
                "*factura*.xlsx",
                "*factura*.xls",
                "*factura*.csv",
                "*casagres*.xlsx",
                "*icmd*.xlsx",
                "*.parquet",
                "*.xlsx",
                "*.xls",
                "*.csv",
            ),
        )

    frame = _read_table(path, sheet_name=sheet_name)

    date_column = date_column or _column_by_candidates(
        frame,
        (
            "Periodo",
            "Fecha",
            "Date",
            "Mes",
            "AñoMes",
            "YearMonth",
        ),
        required=True,
    )
    target_column = target_column or _column_by_candidates(
        frame,
        (
            "Valor Venta",
            "ValorVenta",
            "valor_venta",
            "SalesValue",
            "Valor Neto",
            "Venta Neta",
        ),
        required=True,
    )
    series_id_column = series_id_column or _column_by_candidates(
        frame,
        (
            "Material",
            "Referencia",
            "Producto",
            "Codigo Material",
            "Código Material",
            "Product",
            "ProductID",
        ),
        required=False,
    )

    working = frame.copy()
    working["_month"] = _parse_month_column(working[date_column])
    working["_target"] = _parse_locale_number(working[target_column])
    working = working.dropna(subset=["_month", "_target"])

    if positive_sales_only:
        working = working[working["_target"] > 0.0]

    if series_id_column is None:
        working["_series_id"] = "ICMD"
        series_id_column = "_series_id"
    else:
        working[series_id_column] = working[series_id_column].astype(str)

    if selected_series is not None:
        allowed = {str(value) for value in selected_series}
        working = working[working[series_id_column].isin(allowed)]

    monthly = (
        working.groupby([series_id_column, "_month"], as_index=False)["_target"]
        .sum()
        .sort_values([series_id_column, "_month"])
    )

    train_map: dict[str, pd.Series] = {}
    test_map: dict[str, pd.Series] = {}
    metadata_rows: list[dict[str, Any]] = []

    for series_id, group in monthly.groupby(series_id_column):
        group = group.sort_values("_month")

        start = pd.Period(group["_month"].min(), freq="M")
        end = pd.Period(group["_month"].max(), freq="M")
        complete_index = pd.period_range(start=start, end=end, freq="M").to_timestamp("M")

        series = pd.Series(
            group["_target"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(group["_month"]),
            name=str(series_id),
        )
        series = series.groupby(level=0).sum().reindex(complete_index)

        if fill_missing_months is None and series.isna().any():
            raise ValueError(
                f"ICMD/{series_id} contiene meses faltantes. "
                "Defina fill_missing_months explícitamente."
            )

        if fill_missing_months is not None:
            series = series.fillna(float(fill_missing_months))

        if len(series) <= int(test_horizon):
            continue

        train_map[str(series_id)] = series.iloc[:-int(test_horizon)].copy()
        test_map[str(series_id)] = series.iloc[-int(test_horizon):].copy()

        metadata_rows.append(
            {
                "dataset": "ICMD",
                "series_id": str(series_id),
                "n_transactions": int(
                    (working[series_id_column] == str(series_id)).sum()
                ),
                "n_months": len(series),
                "forecast_horizon": int(test_horizon),
                "start_timestamp": series.index.min(),
                "end_timestamp": series.index.max(),
            }
        )

    bundle = DatasetBundle(
        name="ICMD",
        train_series_map=train_map,
        test_series_map=test_map,
        metadata=pd.DataFrame(metadata_rows),
        seasonal_period=12,
    )

    return _validate_bundle(
        bundle,
        minimum_train_length=minimum_train_length,
        maximum_train_length=maximum_train_length,
        maximum_series=maximum_series,
        selected_series=selected_series,
    )


# ============================================================
# Loader Series Prioritarias (JSON)
# ============================================================

def load_series_prioritarias(
    dataset_path: str | Path = "series_prioritarias.json",
    *,
    dataset_name: str = "Series_Prioritarias",
    test_horizon: int = 6,
    minimum_train_length: int = 12,
    maximum_train_length: int | None = 199,
    maximum_series: int | None = None,
    selected_series: Sequence[str] | None = None,
    seasonal_period: int = 12,
) -> DatasetBundle:
    """
    Loads time series from a JSON file (format {series_id: {"index": [...], "values": [...]}})
    and returns a standardized DatasetBundle.
    """
    path = Path(dataset_path)

    if path.is_dir():
        path = _find_file(
            path,
            (
                "series_prioritarias.json",
                "*series*.json",
                "*.json",
            ),
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_map: dict[str, pd.Series] = {}
    test_map: dict[str, pd.Series] = {}
    metadata_rows: list[dict[str, Any]] = []

    selected_set = (
        {str(s) for s in selected_series} if selected_series is not None else None
    )

    for series_id, payload in data.items():
        series_id_str = str(series_id)

        if selected_set is not None and series_id_str not in selected_set:
            continue

        raw_index = payload.get("index", [])
        raw_values = payload.get("values", [])

        if not raw_index or not raw_values:
            continue

        index = pd.to_datetime(raw_index)
        values = np.asarray(raw_values, dtype=float)

        series = pd.Series(values, index=index, name=series_id_str).sort_index()

        if len(series) <= int(test_horizon):
            continue

        train = series.iloc[:-int(test_horizon)].copy()
        test = series.iloc[-int(test_horizon):].copy()

        train_map[series_id_str] = train
        test_map[series_id_str] = test

        metadata_rows.append(
            {
                "dataset": dataset_name,
                "series_id": series_id_str,
                "n_total": len(series),
                "n_train": len(train),
                "n_test": len(test),
                "forecast_horizon": int(test_horizon),
                "start_timestamp": series.index.min(),
                "end_timestamp": series.index.max(),
                "seasonal_period": seasonal_period,
            }
        )

    bundle = DatasetBundle(
        name=dataset_name,
        train_series_map=train_map,
        test_series_map=test_map,
        metadata=pd.DataFrame(metadata_rows),
        seasonal_period=seasonal_period,
    )

    return _validate_bundle(
        bundle,
        minimum_train_length=minimum_train_length,
        maximum_train_length=maximum_train_length,
        maximum_series=maximum_series,
        selected_series=selected_series,
    )


# ============================================================
# Dispatcher común
# ============================================================

def load_dataset(
    dataset_name: str,
    dataset_path: str | Path,
    **kwargs: Any,
) -> DatasetBundle:
    key = _normalized_name(dataset_name)

    loaders = {
        "m4": load_m4_monthly,
        "m4monthly": load_m4_monthly,
        "m3": load_m3_monthly,
        "m3monthly": load_m3_monthly,
        "tourism": load_tourism_monthly,
        "tourismmonthly": load_tourism_monthly,
        "cif": load_cif_2016,
        "cif2016": load_cif_2016,
        "icmd": load_icmd_monthly,
        "casagres": load_icmd_monthly,
        "seriesprioritarias": load_series_prioritarias,
        "series_prioritarias": load_series_prioritarias,
        "prioritarias": load_series_prioritarias,
        "json": load_series_prioritarias,
    }

    if key not in loaders:
        raise KeyError(
            f"Dataset desconocido: {dataset_name}. "
            f"Opciones: {sorted(loaders)}"
        )

    return loaders[key](dataset_path, **kwargs)
