from __future__ import annotations

from pathlib import Path

from .utils import load_json


def load_feature_json(path: str | Path) -> dict:
    return load_json(path)


def get_selected_smart_bases(feature_json: dict, disk_model: str) -> list[str]:
    """
    Expected JSON style:
    {
      "ST12000NM0008": ["smart_1", "smart_5", "smart_9", ...],
      "HGST HUH721212ALN604": ["smart_10", "smart_12", ...]
    }

    Returns normalized SMART base names, e.g.:
    ["smart_1", "smart_5", "smart_9"]

    The function also tolerates legacy entries such as "1" and full column
    names such as "smart_1_raw" or "smart_1_normalized".
    """
    if disk_model not in feature_json:
        raise KeyError(f"Disk model '{disk_model}' not found in feature JSON.")

    selected = feature_json[disk_model]
    if not isinstance(selected, list):
        raise ValueError(
            f"Feature JSON entry for '{disk_model}' must be a list, got: {type(selected)}"
        )

    normalized: list[str] = []
    seen: set[str] = set()

    for x in selected:
        s = str(x).strip()
        if not s:
            continue

        # Support full input-column names, e.g., smart_1_raw -> smart_1.
        if s.endswith("_raw"):
            s = s[: -len("_raw")]
        elif s.endswith("_normalized"):
            s = s[: -len("_normalized")]

        # Support legacy pure-id style, e.g., 1 -> smart_1.
        if not s.startswith("smart_"):
            s = f"smart_{s}"

        if s not in seen:
            normalized.append(s)
            seen.add(s)

    if not normalized:
        raise ValueError(f"No valid SMART features found for disk model '{disk_model}'.")

    return normalized


def expand_smart_feature_candidates(smart_bases: list[str]) -> list[str]:
    """
    Convert SMART base names to likely input columns.
    Example:
      smart_10 -> smart_10_normalized, smart_10_raw
    """
    cols: list[str] = []
    seen: set[str] = set()

    for base in smart_bases:
        for col in (f"{base}_normalized", f"{base}_raw"):
            if col not in seen:
                cols.append(col)
                seen.add(col)

    return cols


def filter_existing_columns(candidate_cols: list[str], available_cols: list[str]) -> list[str]:
    """
    Keep candidate columns that exist in the input dataframe while preserving
    candidate order and removing duplicates.
    """
    available_set = set(available_cols)
    filtered: list[str] = []
    seen: set[str] = set()

    for col in candidate_cols:
        if col in available_set and col not in seen:
            filtered.append(col)
            seen.add(col)

    return filtered
