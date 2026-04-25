
#!/usr/bin/env python3
"""
af3_stats.py — Compute mean and median for PAE, atom_plddts, and contact_prob
from AlphaFold3-style JSON outputs.

Usage:
    python af3_stats.py path/to/file1.json [path/to/file2.json ...]
    python af3_stats.py --out summary.csv file1.json file2.json
    
    For example
    python AF3_stats.py statistics/*.json --out af3_metrics.csv

What it does:
- Recursively searches the JSON dict/list for keys that look like:
  - PAE: 'pae', 'pae_matrix', 'predicted_aligned_error'
  - atom_plddts: 'atom_plddt', 'atom_plddts', 'plddt', 'pLDDT'
  - contact_prob: 'contact_prob', 'contact_probability', 'contact_probs'
- Flattens list/array/matrix values and computes mean & median (ignoring None/NaN).
- Prints results and optionally writes a CSV summary.
"""

import argparse
import json
import math
import os
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple, Union

Number = Union[int, float]

# ---- Configure common aliases (extend if needed) ----
KEY_ALIASES: Dict[str, List[str]] = {
    "pae": [
        "pae", "pae_matrix", "predicted_aligned_error",
        "PAE", "PredictedAlignedError"
    ],
    "atom_plddts": [
        "atom_plddt", "atom_plddts", "plddt", "pLDDT",
        "per_atom_plddt", "per_residue_plddt"  # include residue-level if present
    ],
    "contact_prob": [
        "contact_prob", "contact_probability", "contact_probs",
        "predicted_contacts", "contact_matrix"
    ],
}

# ---- Utility functions ----

def is_number(x: Any) -> bool:
    """Return True if x is a finite number."""
    return isinstance(x, (int, float)) and not math.isnan(x) and math.isfinite(x)

def flatten(values: Any) -> List[Number]:
    """
    Flatten nested lists/tuples of numbers and filter only finite numbers.
    Accepts None, scalars, or nested containers.
    """
    out: List[Number] = []
    def _walk(v: Any):
        if v is None:
            return
        if isinstance(v, (list, tuple)):
            for item in v:
                _walk(item)
        elif is_number(v):
            out.append(v)  # keep finite numbers
        # ignore dicts or other types here; extraction will provide arrays
    _walk(values)
    return out

def safe_stats(values: List[Number]) -> Tuple[Optional[float], Optional[float]]:
    """Compute mean and median for a list, returning (mean, median) or (None, None) if empty."""
    if not values:
        return None, None
    # statistics.mean/median work on floats/ints
    return float(mean(values)), float(median(values))

def recursive_find(data: Any, target_keys: List[str]) -> List[Any]:
    """
    Recursively search dict/list structures for values under any of the target_keys.
    Returns a list of raw value objects (could be arrays/matrices/scalars).
    """
    found: List[Any] = []

    def _search(obj: Any):
        if isinstance(obj, dict):
            for k, v in obj.items():
                # match if key name equals any alias (case-insensitive)
                if any(k.lower() == alias.lower() for alias in target_keys):
                    found.append(v)
                # continue recursion
                _search(v)
        elif isinstance(obj, list):
            for item in obj:
                _search(item)
        # ignore other types
    _search(data)
    return found

def extract_metric(data: Any, aliases: List[str]) -> List[Number]:
    """
    Using aliases, find metric arrays/matrices and return flattened numeric list.
    If multiple matches are found, concatenate them.
    """
    raw_values = recursive_find(data, aliases)
    flattened: List[Number] = []
    for rv in raw_values:
        flattened.extend(flatten(rv))
    return flattened

def compute_af3_metrics(json_obj: Any) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Extract and compute metrics from a JSON object.
    Returns:
      {
        "pae": {"mean": x, "median": y, "count": n},
        "atom_plddts": {"mean": x, "median": y, "count": n},
        "contact_prob": {"mean": x, "median": y, "count": n},
      }
    Missing metrics will have None for mean/median and count = 0.
    """
    results: Dict[str, Dict[str, Optional[float]]] = {}

    for metric_name, aliases in KEY_ALIASES.items():
        values = extract_metric(json_obj, aliases)
        m, med = safe_stats(values)
        results[metric_name] = {
            "mean": m,
            "median": med,
            "count": float(len(values))  # count may be helpful
        }
    return results

# ---- CLI ----

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute mean/median for PAE, atom_plddts, and contact_prob from AlphaFold3 JSON."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Path(s) to AlphaFold3 JSON files."
    )
    parser.add_argument(
        "--out",
        dest="out_csv",
        default=None,
        help="Optional path to write a CSV summary."
    )
    return parser.parse_args()

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def to_csv_line(record: Dict[str, Any]) -> str:
    """
    Convert a summary dict to a CSV line.
    Columns:
      file, pae_mean, pae_median, pae_count, atom_plddts_mean, atom_plddts_median, atom_plddts_count,
      contact_prob_mean, contact_prob_median, contact_prob_count
    """
    cols = [
        record.get("file", ""),
        record["pae"].get("mean"),
        record["pae"].get("median"),
        record["pae"].get("count"),
        record["atom_plddts"].get("mean"),
        record["atom_plddts"].get("median"),
        record["atom_plddts"].get("count"),
        record["contact_prob"].get("mean"),
        record["contact_prob"].get("median"),
        record["contact_prob"].get("count"),
    ]
    def fmt(x):
        if x is None:
            return ""
        if isinstance(x, float):
            return f"{x:.6g}"
        return str(x)
    return ",".join(fmt(c) for c in cols)

def main():
    args = parse_args()
    summaries: List[Dict[str, Any]] = []
    for path in args.files:
        try:
            data = load_json(path)
        except Exception as e:
            print(f"[ERROR] Failed to load JSON: {path} — {e}")
            continue

        metrics = compute_af3_metrics(data)
        record = {"file": os.path.basename(path), **metrics}
        summaries.append(record)

        # Pretty print per-file summary
        print(f"\nFile: {path}")
        for k in ["pae", "atom_plddts", "contact_prob"]:
            m = metrics[k]["mean"]
            med = metrics[k]["median"]
            cnt = metrics[k]["count"]
            print(f"  {k}: count={int(cnt)}"
                  f"{', mean=' + f'{m:.6g}' if m is not None else ', mean=NA'}"
                  f"{', median=' + f'{med:.6g}' if med is not None else ', median=NA'}")

    # Optional CSV output
    if args.out_csv:
        header = ",".join([
            "file",
            "pae_mean", "pae_median", "pae_count",
            "atom_plddts_mean", "atom_plddts_median", "atom_plddts_count",
            "contact_prob_mean", "contact_prob_median", "contact_prob_count",
        ])
        try:
            with open(args.out_csv, "w", encoding="utf-8") as f:
                f.write(header + "\n")
                for rec in summaries:
                    f.write(to_csv_line(rec) + "\n")
            print(f"\n[OK] Summary written to: {args.out_csv}")
        except Exception as e:
            print(f"[ERROR] Failed to write CSV '{args.out_csv}': {e}")

if __name__ == "__main__":
    main()