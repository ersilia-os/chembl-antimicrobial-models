import string
from collections import Counter

import pandas as pd

_LABEL_MAP = {"A": "individual", "B": "individual", "M": "merged", "G": "general"}

_AGGREGATE_NAME_MAP = {
    "G_ORG_DR": "general_dose_response",
    "G_ORG_SP": "general_single_point",
}


def _idx_to_suffix(n: int) -> str:
    result = ""
    n += 1
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        result = string.ascii_lowercase[remainder] + result
    return result


def compute_model_name(meta: pd.DataFrame, task_id: int) -> str:
    """Return the human-readable model name for row task_id of the metadata DataFrame."""
    row = meta.iloc[task_id]
    pathogen = row["pathogen"]
    pathogen_meta = meta[meta["pathogen"] == pathogen].reset_index()

    def _base(r: pd.Series) -> str:
        if r.get("source") == "pubchem":
            return "pubchem"
        activity_type = r.get("activity_type")
        if pd.isna(activity_type):
            return _AGGREGATE_NAME_MAP.get(r["name"], r["name"].lower())
        parts = [_LABEL_MAP[r["label"]], activity_type.lower()]
        if int(r["decoys"]) > 0:
            parts.append("decoys")
        return "_".join(parts)

    base_names = [_base(r) for _, r in pathogen_meta.iterrows()]
    counts = Counter(base_names)
    seen: dict[str, int] = {}
    final_names = []
    for bn in base_names:
        if counts[bn] > 1:
            idx = seen.get(bn, 0)
            final_names.append(f"{bn}_{_idx_to_suffix(idx)}")
            seen[bn] = idx + 1
        else:
            final_names.append(bn)

    pos = pathogen_meta.index[pathogen_meta["name"] == row["name"]][0]
    return final_names[pos]
