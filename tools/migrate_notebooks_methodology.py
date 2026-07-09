"""Migrate notebooks to the current prior/HPO methodology contract."""

from __future__ import annotations

import json
import re
from pathlib import Path


NOTEBOOK_DIR = Path("notebooks")
CONTRACT_MARKER = "# Methodology contract smoke check"


CONTRACT_SOURCE = [
    CONTRACT_MARKER + "\n",
    "from s3paper.priors import MANDATORY_PRIOR_NAMES, OPTIONAL_PRIOR_NAMES, PRIOR_REGISTRY\n",
    "\n",
    "required_priors = set(MANDATORY_PRIOR_NAMES) | set(OPTIONAL_PRIOR_NAMES)\n",
    "missing_priors = required_priors.difference(PRIOR_REGISTRY)\n",
    "if missing_priors:\n",
    "    raise AssertionError(f\"Missing required priors: {sorted(missing_priors)}\")\n",
    "\n",
    "print(\"Prior registry OK:\", sorted(required_priors))\n",
]


REPLACEMENTS = {
    "oob_split_ratio": "calibration_split_ratio",
    "rolling_mean": "causal_rolling_mean",
}


OLD_WINDOW_LOOKUP = 'S3_POINT_PARAMS[\n                "foundation_window"\n            ]'
NEW_WINDOW_LOOKUP = (
    'S3_POINT_PARAMS.get(\n'
    '                "prior__window",\n'
    '                S3_POINT_PARAMS.get("foundation_window", 6),\n'
    "            )"
)


LOCAL_CONFORMAL_PATTERN = re.compile(
    r"def finite_sample_conformal_quantile\([\s\S]*?\):\n.*?(?=\n\n\ndef adaptive_conformal_path)",
    flags=re.DOTALL,
)


CANONICAL_CONFORMAL_SOURCE = (
    "def finite_sample_conformal_quantile(scores, alpha):\n"
    "    from s3paper.conformal import finite_sample_quantile\n"
    "\n"
    "    return finite_sample_quantile(scores, alpha)\n"
)


def migrate_notebook(path: Path) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    cells = data.get("cells", [])
    changed = False
    has_contract = any(CONTRACT_MARKER in "".join(c.get("source", [])) for c in cells)
    insert_at = None

    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        original = source
        for old, new in REPLACEMENTS.items():
            source = source.replace(old, new)
        source = source.replace(OLD_WINDOW_LOOKUP, NEW_WINDOW_LOOKUP)
        source = LOCAL_CONFORMAL_PATTERN.sub(CANONICAL_CONFORMAL_SOURCE, source)
        if source != original:
            cell["source"] = source.splitlines(keepends=True)
            changed = True
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True
        if insert_at is None and "set_global_seed(SEED)" in source:
            insert_at = i + 1

    if not has_contract:
        cells.insert(
            insert_at or 0,
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": CONTRACT_SOURCE,
            },
        )
        changed = True

    if changed:
        path.write_bytes((json.dumps(data, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    return changed


def main() -> None:
    changed = [path.name for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")) if migrate_notebook(path)]
    for name in changed:
        print(name)


if __name__ == "__main__":
    main()
