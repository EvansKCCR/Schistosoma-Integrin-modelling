# How to run the script
"python AF3_stats.py --input_dir models --output *_af3_metrics.xlsx" 
"--input_dir  directory containing AlphaFold confidence JSON files"
"--output name of Excel file that will be created"
"--glob pattern for files (default *.json)"

import os
import json
import glob
import hashlib
import argparse
from typing import List, Tuple, Dict, Optional

import pandas as pd


# ============================================================
# JSON LOADING & VALIDATION
# ============================================================

def safe_load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to parse JSON: {path} -> {e}")
        return None


def is_valid_af3_schema(data: dict) -> bool:
    """
    Minimal schema validation for AF3 confidence JSON.
    Ensures at least one expected AF3 metric exists.
    """
    expected_keys = {
        "ptm",
        "iptm",
        "ranking_score",
        "chain_iptm",
        "chain_ptm",
        "chain_pair_iptm",
        "chain_pair_pae_min",
    }
    return any(k in data for k in expected_keys)


# ============================================================
# MODEL ID HANDLING
# ============================================================

def generate_model_id(path: str) -> str:
    """
    Generate a stable, unique model_id from filename.
    Uses filename + short hash to avoid collisions.
    """
    base = os.path.splitext(os.path.basename(path))[0]
    digest = hashlib.md5(base.encode("utf-8")).hexdigest()[:6]
    return f"{base}_{digest}"


# ============================================================
# DATAFRAME BUILDERS
# ============================================================

def make_summary_df(model_id: str, af3: dict, source_file: str) -> pd.DataFrame:
    rows = [
        ("ptm", af3.get("ptm")),
        ("iptm", af3.get("iptm")),
        ("ranking_score", af3.get("ranking_score")),
        ("fraction_disordered", af3.get("fraction_disordered")),
        ("has_clash", af3.get("has_clash")),
        ("num_recycles", af3.get("num_recycles")),
    ]

    df = pd.DataFrame(rows, columns=["metric", "value"])
    df["model_id"] = model_id
    df["source_file"] = os.path.basename(source_file)
    return df


def make_per_chain_df(model_id: str, af3: dict, source_file: str) -> pd.DataFrame:
    chain_iptm = af3.get("chain_iptm", [])
    chain_ptm = af3.get("chain_ptm", [])
    n = max(len(chain_iptm), len(chain_ptm))

    if n == 0:
        return pd.DataFrame()

    def pad(lst, n_):
        return (lst + [None] * (n_ - len(lst))) if lst else [None] * n_

    df = pd.DataFrame({
        "model_id": [model_id] * n,
        "source_file": [os.path.basename(source_file)] * n,
        "chain": [f"Chain {i+1}" for i in range(n)],
        "chain_iptm": pad(chain_iptm, n),
        "chain_ptm": pad(chain_ptm, n),
    })
    return df


def matrix_to_df(matrix: Optional[List[List[float]]],
                 model_id: str,
                 source_file: str,
                 value_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if not matrix:
        return pd.DataFrame(), pd.DataFrame()

    n = len(matrix)
    labels = [f"Chain {i+1}" for i in range(n)]

    df = pd.DataFrame(matrix, index=labels, columns=labels)
    df.index.name = "row_chain"
    df.columns.name = "col_chain"

    long_df = (
        df.reset_index()
        .melt(id_vars="row_chain", var_name="col_chain", value_name=value_name)
    )

    long_df.insert(0, "model_id", model_id)
    long_df.insert(1, "source_file", os.path.basename(source_file))

    return df, long_df


def safe_sheet_name(name: str) -> str:
    cleaned = name.replace("/", "_").replace("\\", "_")
    return cleaned[:31]


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export AlphaFold3 JSON metrics to Excel (production version)."
    )
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing AF3 JSON files.")
    parser.add_argument("--output", default="af3_metrics.xlsx",
                        help="Output Excel filename.")
    parser.add_argument("--glob", default="*.json",
                        help="Glob pattern for JSON files (default: *.json)")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, args.glob)))

    if not files:
        print("[ERROR] No JSON files found.")
        return

    print(f"[INFO] Found {len(files)} JSON files.")
    print(f"[INFO] Writing Excel output -> {args.output}")

    all_summary = []
    all_per_chain = []
    all_pair_iptm = []
    all_pair_pae = []

    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:

        for path in files:
            data = safe_load_json(path)
            if data is None:
                continue

            if not is_valid_af3_schema(data):
                print(f"[WARN] Skipping non-AF3 JSON: {path}")
                continue

            model_id = generate_model_id(path)

            # Summary
            summary_df = make_summary_df(model_id, data, path)
            summary_df.to_excel(
                writer,
                sheet_name=safe_sheet_name(f"{model_id}_summary"),
                index=False,
            )
            all_summary.append(summary_df)

            # Per-chain
            per_chain_df = make_per_chain_df(model_id, data, path)
            if not per_chain_df.empty:
                per_chain_df.to_excel(
                    writer,
                    sheet_name=safe_sheet_name(f"{model_id}_per_chain"),
                    index=False,
                )
                all_per_chain.append(per_chain_df)

            # Pairwise matrices
            iptm_mat = data.get("chain_pair_iptm")
            pae_mat = data.get("chain_pair_pae_min")

            pair_iptm_df, pair_iptm_long = matrix_to_df(
                iptm_mat, model_id, path, "pair_iptm"
            )
            pair_pae_df, pair_pae_long = matrix_to_df(
                pae_mat, model_id, path, "pair_pae_min"
            )

            if not pair_iptm_df.empty:
                pair_iptm_df.to_excel(
                    writer,
                    sheet_name=safe_sheet_name(f"{model_id}_iptm_matrix"),
                )
                all_pair_iptm.append(pair_iptm_long)

            if not pair_pae_df.empty:
                pair_pae_df.to_excel(
                    writer,
                    sheet_name=safe_sheet_name(f"{model_id}_pae_matrix"),
                )
                all_pair_pae.append(pair_pae_long)

        # ====================================================
        # AGGREGATED SHEETS
        # ====================================================

        if all_summary:
            summary_all = pd.concat(all_summary, ignore_index=True)
            summary_pivot = (
                summary_all
                .pivot_table(
                    index=["model_id", "source_file"],
                    columns="metric",
                    values="value",
                    aggfunc="first",
                )
                .reset_index()
            )
            summary_pivot.to_excel(writer, sheet_name="ALL_summary", index=False)

        if all_per_chain:
            pd.concat(all_per_chain, ignore_index=True)\
                .to_excel(writer, sheet_name="ALL_per_chain", index=False)

        if all_pair_iptm:
            pd.concat(all_pair_iptm, ignore_index=True)\
                .to_excel(writer, sheet_name="ALL_pair_iptm_long", index=False)

        if all_pair_pae:
            pd.concat(all_pair_pae, ignore_index=True)\
                .to_excel(writer, sheet_name="ALL_pair_pae_long", index=False)

    print("[DONE] Excel export completed successfully.")


if __name__ == "__main__":
    main()