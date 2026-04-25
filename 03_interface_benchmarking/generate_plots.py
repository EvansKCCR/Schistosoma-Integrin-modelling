#!/usr/bin/env python3
# ============================================================== 
# Standalone Plot Generator for Composite Scoring Pipeline
# Usage:
#     python generate_plots.py composite_scores_with_sensitivity.xlsx
# If no filename is provided, defaults to: composite_scores_with_sensitivity.xlsx
# ==============================================================

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")

# --------------------------------------------------------------
# 0) Read input
# --------------------------------------------------------------
if len(sys.argv) > 1:
    infile = sys.argv[1]
else:
    infile = "composite_scores_with_sensitivity.xlsx"

print(f"\nLoading: {infile}")

try:
    summary = pd.read_excel(infile, sheet_name="scenario_scores_ranks")
    norm = pd.read_excel(infile, sheet_name="normalized_metrics")
    raw = pd.read_excel(infile, sheet_name="original_input")
except Exception as e:
    print("❌ ERROR: File must contain sheets: scenario_scores_ranks, normalized_metrics, original_input")
    print("Details:", e)
    sys.exit(1)

# Identify useful items
if "model" in summary.columns:
    model_labels = summary["model"]
else:
    model_labels = summary.index.astype(str)

metrics = [c for c in norm.columns if c not in ["model", "template", "template_name"]]
score_cols = [c for c in summary.columns if c.startswith("score__")]
rank_cols = [c for c in summary.columns if c.startswith("rank__")]

# Make plot directory
os.makedirs("plots", exist_ok=True)

# --------------------------------------------------------------
# 1) Score Distributions
# --------------------------------------------------------------
def plot_score_distributions():
    for c in score_cols:
        plt.figure(figsize=(7,5))
        sns.histplot(summary[c], kde=True, color="royalblue")
        plt.title(f"Distribution of {c}")
        plt.xlabel("Score")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"plots/{c}.png", dpi=300)
        plt.close()

# --------------------------------------------------------------
# 2) Rank Heatmap
# --------------------------------------------------------------
def plot_rank_heatmap():
    if not rank_cols:
        print("⚠ Rank heatmap skipped (no rank__ columns found).")
        return
    df = summary[rank_cols].set_index(model_labels)
    plt.figure(figsize=(12, 6))
    sns.heatmap(df, annot=True, cmap="viridis_r")
    plt.title("Rank Comparison Across Scenarios")
    plt.tight_layout()
    plt.savefig("plots/rank_heatmap.png", dpi=300)
    plt.close()

# --------------------------------------------------------------
# 3) Pareto Front Plot
# --------------------------------------------------------------
def plot_pareto_front():
    needed = ["score__af_confidence", "score__docking_interface"]
    if not all(col in summary.columns for col in needed):
        print("⚠ Pareto plot skipped (missing AF or docking scores).")
        return

    colors = summary.get("pareto_non_dominated", pd.Series([False]*len(summary))).map({True: "red", False: "gray"})

    plt.figure(figsize=(7,5))
    plt.scatter(summary["score__af_confidence"],
                summary["score__docking_interface"],
                c=colors, s=90)
    plt.xlabel("AF Confidence Score")
    plt.ylabel("Docking Interface Score")
    plt.title("Pareto Front (Red = Non-Dominated Models)")
    plt.tight_layout()
    plt.savefig("plots/pareto_front.png", dpi=300)
    plt.close()

# --------------------------------------------------------------
# 4) Radar Plots for Recommended Models
# --------------------------------------------------------------
def plot_radar():
    if "recommended" not in summary.columns or "model" not in summary.columns:
        print("⚠ Radar plots skipped (need 'recommended' and 'model' columns).")
        return

    recommended = summary[summary["recommended"] == True]["model"]

    for m in recommended:
        # Locate row index in summary and norm
        try:
            idx = summary.index[summary["model"] == m][0]
        except IndexError:
            print(f"  ! Could not locate model in summary: {m}")
            continue
        vals = norm.loc[idx, metrics].values
        labels = metrics

        # Radar setup
        angles = np.linspace(0, 2*np.pi, len(vals), endpoint=False)
        vals = np.concatenate([vals, [vals[0]]])
        angles = np.concatenate([angles, [angles[0]]])

        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, vals, "o-", linewidth=2)
        ax.fill(angles, vals, alpha=0.25)
        ax.set_thetagrids(angles[:-1] * 180/np.pi, labels)
        ax.set_title(f"Radar Plot: {m}", y=1.08)
        plt.tight_layout()
        plt.savefig(f"plots/radar_{m}.png", dpi=300)
        plt.close()

# --------------------------------------------------------------
# 5) Sensitivity OAT Plots
# --------------------------------------------------------------
def plot_sensitivity_OAT():
    try:
        oat = pd.read_excel(infile, sheet_name="sensitivity_OAT")
    except Exception:
        print("⚠ No OAT sensitivity sheet found.")
        return

    scenarios = oat["scenario"].dropna().unique()
    for sc in scenarios:
        sub = oat[oat["scenario"] == sc]
        if sub.empty:
            continue
        plt.figure(figsize=(10,5))
        sns.barplot(
            data=sub,
            x="metric_perturbed",
            y="spearman_rho",
            hue="direction",
            palette="coolwarm"
        )
        plt.ylim(0, 1)
        plt.title(f"Sensitivity Analysis (OAT): {sc}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"plots/OAT_{sc}.png", dpi=300)
        plt.close()

# --------------------------------------------------------------
# 6) Sensitivity Monte-Carlo Plots
# --------------------------------------------------------------
def plot_sensitivity_MC():
    try:
        mc = pd.read_excel(infile, sheet_name="sensitivity_MC_topN")
    except Exception:
        print("⚠ No Monte-Carlo sensitivity sheet found.")
        return

    scenarios = mc["scenario"].dropna().unique()
    for sc in scenarios:
        sub = mc[mc["scenario"] == sc]
        if sub.empty:
            continue
        plt.figure(figsize=(10,5))
        sns.barplot(data=sub, x="model", y="topN_freq")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Top-N Frequency")
        plt.title(f"Monte-Carlo Sensitivity: {sc}")
        plt.tight_layout()
        plt.savefig(f"plots/MC_{sc}.png", dpi=300)
        plt.close()

# --------------------------------------------------------------
# RUN ALL PLOTS
# --------------------------------------------------------------
print("Generating plots...")
plot_score_distributions()
plot_rank_heatmap()
plot_pareto_front()
plot_radar()
plot_sensitivity_OAT()
plot_sensitivity_MC()
print("\nAll plots saved to: ./plots/\n")
