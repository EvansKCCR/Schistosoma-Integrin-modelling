#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Excel summary + all high-resolution plots from Desmond_generated.xlsx.

Outputs:
- SmInt1_MD_summary.xlsx (stats tables incl. t_eq-based windows)
- Fig1_RMSD_timecourse.png
- Fig2_LigandRMSD_timecourse.png
- Fig3_LigandRMSD_distributions.png
- Fig4_rGyr_timecourse.png
- Fig5_SurfaceMetrics_RGD_vs_Mut.png
- Fig6_DeltaRMSF.png
- Fig7_PLContacts_PosCtrl.png
- Fig8_PLContacts_RGD.png
- Fig9_PLContacts_Mutated.png
- Fig10_PLContacts_ComparativeLollipop.png
- Fig11_PLContacts_ProxyFractionBound.png
- Fig12_PLContacts_ProxyIndex.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LinearRegression
from scipy.stats import sem

# -----------------------------
# Global plotting parameters (600 dpi output)
# -----------------------------
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 600,      # <-- high resolution
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PALETTE = {
    "Pos control": "#1f77b4",
    "RGD": "#2ca02c",
    "Mutated": "#d62728",
    "Non RGD": "#9467bd",
    "APO": "#8c564b",
}

EXCEL_IN = "trajectory_data.xlsx"
SUMMARY_OUT = "MD_summary.xlsx"

# -----------------------------
# Helpers
# -----------------------------
def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def detect_teq(time_ns: pd.Series, values: pd.Series,
               window_ns: float = 5.0, tol: float = 1.0) -> float:
    """
    Detect equilibration time: first index where a rolling window (window_ns)
    of Cα-RMSD has max-min <= tol. If none, return 20 ns.
    """
    t = to_num(time_ns).dropna().reset_index(drop=True)
    v = to_num(values).dropna().reset_index(drop=True)
    if len(t) < 2:
        return float(t.iloc[0]) if len(t) else 0.0
    dt = np.median(np.diff(t))
    win = max(1, int(round(window_ns / dt)))
    roll = pd.Series(v.values).rolling(win, min_periods=win).apply(
        lambda x: np.max(x) - np.min(x), raw=True
    )
    idx = None
    for i, spread in enumerate(roll):
        if i >= win - 1 and spread <= tol:
            idx = i - (win - 1)
            break
    return float(t.iloc[idx]) if idx is not None else float(t.iloc[0] + 20.0)

def mean_ci(series: pd.Series):
    a = to_num(series).dropna().values
    if a.size == 0:
        return np.nan, np.nan, np.nan
    m = float(np.mean(a))
    sd = float(np.std(a, ddof=1)) if a.size > 1 else 0.0
    se = sem(a) if a.size > 1 else 0.0
    return m, sd, 1.96 * se

def round_df(df, ndigits=3):
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(ndigits)
    return out

def savefig(path, fig=None):
    if fig is not None:
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

# -----------------------------
# Load workbook
# -----------------------------
xl = pd.ExcelFile(EXCEL_IN, engine="openpyxl")

# Sheets (as inspected previously in your file)
rmsd = xl.parse("C_alpha_RMSD")
lig = xl.parse("Lig_RMSD")
rgyr = xl.parse("rGyr")
sasa = xl.parse("SASA2")
molsa = xl.parse("MolSA2")
psa = xl.parse("PSA_")
hb = xl.parse("intraHB")
rmsf = xl.parse("RMSF")
pl_contacts = xl.parse("P_L_contact")

# Coerce numerics for data sheets
for df in [rmsd, lig, rgyr, sasa, molsa, psa, hb, rmsf]:
    for c in df.columns:
        if c not in ["Frame", "Time (ns)"]:
            df[c] = to_num(df[c])

# Drop empty unnamed columns from PSA
psa = psa.loc[:, ~psa.columns.str.contains("^Unnamed")]

# -----------------------------
# Equilibration detection from protein RMSD
# -----------------------------
cond_rmsd = [c for c in rmsd.columns if c not in ["Frame", "Time (ns)"]]
teq = {c: detect_teq(rmsd["Time (ns)"], rmsd[c]) for c in cond_rmsd}

# -----------------------------
# ========== Excel summary ==========
# -----------------------------
def build_rmsd_stats():
    rows = []
    for c in cond_rmsd:
        t = rmsd["Time (ns)"]
        v = rmsd[c]
        mean_all, sd_all, ci_all = mean_ci(v)
        pct2p5_all = float((v <= 2.5).mean() * 100.0)
        t0 = teq[c]
        veq = v[t >= t0]
        mean_eq, sd_eq, ci_eq = mean_ci(veq)
        pct2p5_eq = float((veq <= 2.5).mean() * 100.0) if veq.size > 0 else np.nan
        rows.append({
            "Condition": c, "t_eq (ns)": t0,
            "RMSD mean (0-100)": mean_all, "RMSD SD (0-100)": sd_all,
            "RMSD 95%CI half": ci_all, "% frames ≤2.5Å (0-100)": pct2p5_all,
            "RMSD mean (eq)": mean_eq, "RMSD SD (eq)": sd_eq,
            "RMSD 95%CI half (eq)": ci_eq, "% frames ≤2.5Å (eq)": pct2p5_eq
        })
    return pd.DataFrame(rows)

def build_ligand_stats():
    conds = [c for c in lig.columns if c not in ["Frame", "Time (ns)"]]
    rows = []
    for c in conds:
        t = lig["Time (ns)"]
        v = lig[c]
        med = float(v.median())
        p90 = float(v.quantile(0.9))
        pct2 = float((v <= 2.0).mean() * 100.0)
        pct3 = float((v <= 3.0).mean() * 100.0)
        pct4 = float((v <= 4.0).mean() * 100.0)
        dt = np.median(np.diff(t.dropna())) if t.dropna().size > 1 else 0.1
        need = max(1, int(round(2.0 / dt)))
        above = (v > 5.0).astype(int).fillna(0)
        runmax = 0; cur = 0
        for val in above:
            if val == 1:
                cur += 1; runmax = max(runmax, cur)
            else:
                cur = 0
        sustained = bool(runmax >= need)
        t0 = teq.get(c, min(teq.values()))
        v2 = v[t >= t0]
        med_eq = float(v2.median()) if v2.size > 0 else np.nan
        p90_eq = float(v2.quantile(0.9)) if v2.size > 0 else np.nan
        pct2_eq = float((v2 <= 2.0).mean() * 100.0) if v2.size > 0 else np.nan
        pct3_eq = float((v2 <= 3.0).mean() * 100.0) if v2.size > 0 else np.nan
        pct4_eq = float((v2 <= 4.0).mean() * 100.0) if v2.size > 0 else np.nan
        rows.append({
            "Condition": c, "t_eq (ns)": t0,
            "Median (0-100)": med, "P90 (0-100)": p90,
            "%≤2Å (0-100)": pct2, "%≤3Å (0-100)": pct3, "%≤4Å (0-100)": pct4,
            "Sustained >5Å ≥2ns?": sustained,
            "Median (eq)": med_eq, "P90 (eq)": p90_eq,
            "%≤2Å (eq)": pct2_eq, "%≤3Å (eq)": pct3_eq, "%≤4Å (eq)": pct4_eq
        })
    return pd.DataFrame(rows)

def build_rgyr_stats():
    conds = [c for c in rgyr.columns if c not in ["Frame", "Time (ns)"]]
    rows = []
    for c in conds:
        t = to_num(rgyr["Time (ns)"])
        v = to_num(rgyr[c])
        mean_all, sd_all, ci_all = mean_ci(v)
        cv_all = float(sd_all / mean_all * 100.0) if mean_all else np.nan
        t_end = t.max()
        mask50 = t >= (t_end - 50.0)
        X = t[mask50].dropna().values.reshape(-1, 1)
        y = v[mask50].dropna().values
        slope = np.nan
        if X.size > 1 and y.size == X.size:
            lr = LinearRegression().fit(X, y)
            slope = float(lr.coef_[0])
        t0 = teq.get(c, min(teq.values()))
        veq = v[t >= t0]
        mean_eq, sd_eq, ci_eq = mean_ci(veq)
        cv_eq = float(sd_eq / mean_eq * 100.0) if mean_eq else np.nan
        rows.append({
            "Condition": c, "t_eq (ns)": t0,
            "rGyr mean (0-100)": mean_all, "rGyr SD (0-100)": sd_all,
            "rGyr %CV (0-100)": cv_all, "Slope last 50 ns (Å/ns)": slope,
            "rGyr mean (eq)": mean_eq, "rGyr SD (eq)": sd_eq, "rGyr %CV (eq)": cv_eq
        })
    return pd.DataFrame(rows)

def build_scalar_table(df, name):
    cols = [c for c in df.columns if c not in ["Frame", "Time (ns)"]]
    rows = []
    t = df["Time (ns)"] if "Time (ns)" in df.columns else None
    for c in cols:
        m_all, sd_all, _ = mean_ci(df[c])
        if t is not None:
            t0 = teq.get(c, min(teq.values()))
            veq = df[c][t >= t0]
            m_eq, sd_eq, _ = mean_ci(veq)
        else:
            t0, m_eq, sd_eq = (np.nan, np.nan, np.nan)
        rows.append({"Condition": c, "t_eq (ns)": t0,
                     f"{name} mean (0-100)": m_all, f"{name} SD (0-100)": sd_all,
                     f"{name} mean (eq)": m_eq, f"{name} SD (eq)": sd_eq})
    out = pd.DataFrame(rows)
    d_all = d_eq = np.nan
    if "RGD" in out["Condition"].values and "Mutated" in out["Condition"].values:
        rgd_row = out.set_index("Condition").loc["RGD"]
        mut_row = out.set_index("Condition").loc["Mutated"]
        d_all = float(rgd_row[f"{name} mean (0-100)"] - mut_row[f"{name} mean (0-100)"])
        d_eq  = float(rgd_row[f"{name} mean (eq)"] - mut_row[f"{name} mean (eq)"])
    return out, d_all, d_eq

# Build and write summary workbook
rmsd_tbl = round_df(build_rmsd_stats())
lig_tbl  = round_df(build_ligand_stats())
rgyr_tbl = round_df(build_rgyr_stats())

sasa_tbl, sasa_d_all, sasa_d_eq   = build_scalar_table(sasa, "SASA")
molsa_tbl, molsa_d_all, molsa_d_eq = build_scalar_table(molsa, "MolSA")
psa_tbl, psa_d_all, psa_d_eq      = build_scalar_table(psa, "PSA")
sasa_tbl, molsa_tbl, psa_tbl = round_df(sasa_tbl), round_df(molsa_tbl), round_df(psa_tbl)

hb_cols = [c for c in hb.columns if c not in ["Frame", "Time (ns)"]]
hb_rows = []
for c in hb_cols:
    t0 = teq.get(c, min(teq.values()))
    m_all, _, _ = mean_ci(hb[c])
    veq = hb[c][hb["Time (ns)"] >= t0]
    m_eq, _, _ = mean_ci(veq)
    hb_rows.append({"Condition": c, "t_eq (ns)": t0,
                    "intraHB mean (0-100)": m_all, "intraHB mean (eq)": m_eq})
hb_tbl = round_df(pd.DataFrame(hb_rows))

# RMSF means and deltas
res_col = rmsf.columns[0]
for c in rmsf.columns:
    if c != res_col:
        rmsf[c] = to_num(rmsf[c])
rmsf_means = rmsf[[c for c in rmsf.columns if c != res_col]].mean(numeric_only=True)
rmsf_means_tbl = round_df(pd.DataFrame({"Metric": "Mean RMSF (Å)", **{k:[v] for k,v in rmsf_means.items()}}))
rmsf_top_tbl = pd.DataFrame()
if "Mutated" in rmsf.columns and "RGD" in rmsf.columns:
    rmsf["Delta(Mutated-RGD)"] = rmsf["Mutated"] - rmsf["RGD"]
    rmsf_top_tbl = round_df(rmsf[[res_col, "RGD", "Mutated", "Delta(Mutated-RGD)"]]
                            .sort_values("Delta(Mutated-RGD)", ascending=False).head(25))

# Deltas sheet
delta_sheet = round_df(pd.DataFrame([
    {"Metric": "Δ(RGD−Mutated) SASA", "0–100 ns": sasa_d_all, "eq window": sasa_d_eq},
    {"Metric": "Δ(RGD−Mutated) MolSA", "0–100 ns": molsa_d_all, "eq window": molsa_d_eq},
    {"Metric": "Δ(RGD−Mutated) PSA",  "0–100 ns": psa_d_all,  "eq window": psa_d_eq},
]))

teq_tbl = round_df(pd.DataFrame({"Condition": list(teq.keys()),
                                 "t_eq (ns)": list(teq.values())}))

with pd.ExcelWriter(SUMMARY_OUT, engine="openpyxl") as w:
    teq_tbl.to_excel(w, index=False, sheet_name="Equilibration_times")
    rmsd_tbl.to_excel(w, index=False, sheet_name="Protein_RMSD_stats")
    lig_tbl.to_excel(w, index=False, sheet_name="Ligand_RMSD_stats")
    rgyr_tbl.to_excel(w, index=False, sheet_name="rGyr_stats")
    sasa_tbl.to_excel(w, index=False, sheet_name="SASA_stats")
    molsa_tbl.to_excel(w, index=False, sheet_name="MolSA_stats")
    psa_tbl.to_excel(w, index=False, sheet_name="PSA_stats")
    hb_tbl.to_excel(w, index=False, sheet_name="intraHB_stats")
    delta_sheet.to_excel(w, index=False, sheet_name="RGD_vs_Mutant_deltas")
    rmsf_means_tbl.to_excel(w, index=False, sheet_name="RMSF_means")
    if not rmsf_top_tbl.empty:
        rmsf_top_tbl.to_excel(w, index=False, sheet_name="RMSF_top25_deltas")

print(f"[OK] Wrote summary: {SUMMARY_OUT}")

# -----------------------------
# ========== Plots ==========
# -----------------------------

# ---- Fig1: Protein RMSD vs time with t_eq markers
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for c in cond_rmsd:
    ax.plot(rmsd["Time (ns)"], rmsd[c], label=c, lw=1.4, color=PALETTE.get(c))
    ax.axvline(teq[c], color=PALETTE.get(c), ls="--", lw=0.8, alpha=0.7)
ax.set_xlabel("Time (ns)"); ax.set_ylabel("Cα-RMSD (Å)")
ax.set_title("Protein Cα-RMSD with equilibration markers (t_eq)")
ax.legend(ncol=3, fontsize=8, frameon=False)
ax.yaxis.set_major_locator(MaxNLocator(6))
savefig("Fig1_RMSD_timecourse.png", fig)

# ---- Fig2: Ligand RMSD vs time
conds_lig = [c for c in lig.columns if c not in ["Frame", "Time (ns)"]]
fig, ax = plt.subplots(figsize=(7.2, 3.8))
for c in conds_lig:
    ax.plot(lig["Time (ns)"], lig[c], label=c, lw=1.4, color=PALETTE.get(c))
ax.axhline(5.0, color="gray", lw=1.0, ls="--", label="5 Å threshold")
ax.set_xlabel("Time (ns)"); ax.set_ylabel("Ligand RMSD (Å)")
ax.set_title("Ligand RMSD vs time")
ax.legend(ncol=4, fontsize=8, frameon=False); ax.yaxis.set_major_locator(MaxNLocator(6))
savefig("Fig2_LigandRMSD_timecourse.png", fig)

# ---- Fig3: Ligand RMSD distributions (0–100 ns vs eq window)
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.8), sharey=True)
pos, labels, vals_all = [], [], []
for i, c in enumerate(conds_lig):
    vals = to_num(lig[c]).dropna()
    vals_all.append(vals); pos.append(i+1); labels.append(c)
axes[0].violinplot(vals_all, positions=pos, showextrema=False)
axes[0].boxplot(vals_all, positions=pos, widths=0.2, showfliers=False)
axes[0].set_xticks(pos, labels); axes[0].set_title("0–100 ns")
axes[0].set_ylabel("Ligand RMSD (Å)")

pos2, vals_eq = [], []
for i, c in enumerate(conds_lig):
    t0 = teq.get(c, min(teq.values()))
    vals = to_num(lig.loc[lig["Time (ns)"] >= t0, c]).dropna()
    vals_eq.append(vals); pos2.append(i+1)
axes[1].violinplot(vals_eq, positions=pos2, showextrema=False)
axes[1].boxplot(vals_eq, positions=pos2, widths=0.2, showfliers=False)
axes[1].set_xticks(pos2, labels); axes[1].set_title("Equilibrated window (t ≥ t_eq)")
for ax in axes: ax.yaxis.set_major_locator(MaxNLocator(6))
savefig("Fig3_LigandRMSD_distributions.png")

# ---- Fig4: rGyr with linear drift fit over last 50 ns
fig, ax = plt.subplots(figsize=(7.2, 4.0))
conds_rgyr = [c for c in rgyr.columns if c not in ["Frame", "Time (ns)"]]
for c in conds_rgyr:
    col = PALETTE.get(c)
    t = to_num(rgyr["Time (ns)"]); v = to_num(rgyr[c])
    ax.plot(t, v, label=c, lw=1.2, color=col)
    t_end = t.max(); mask = t >= (t_end - 50.0)
    X = t[mask].values.reshape(-1, 1); y = v[mask].values
    if X.size > 1 and y.size == X.size:
        lr = LinearRegression().fit(X, y); yfit = lr.predict(X)
        ax.plot(t[mask], yfit, color=col, ls="--", lw=1.0, alpha=0.8)
ax.set_xlabel("Time (ns)"); ax.set_ylabel("Radius of gyration (Å)")
ax.set_title("rGyr with linear drift (last 50 ns)")
ax.legend(ncol=3, fontsize=8, frameon=False); ax.yaxis.set_major_locator(MaxNLocator(6))
savefig("Fig4_rGyr_timecourse.png", fig)

# ---- Fig5: Surface metrics (RGD vs Mutated) for SASA/MolSA/PSA
metrics = [("SASA", sasa), ("MolSA", molsa), ("PSA", psa)]
means_all, means_eq = [], []
for name, df in metrics:
    if {"RGD", "Mutated"}.issubset(df.columns):
        t = df["Time (ns)"]
        t0_rgd = teq.get("RGD", 0.0)
        t0_mut = teq.get("Mutated", 0.0)
        means_all.append([name, df["RGD"].mean(), df["Mutated"].mean()])
        means_eq.append([name, df.loc[t >= t0_rgd, "RGD"].mean(),
                               df.loc[t >= t0_mut, "Mutated"].mean()])
means_all = pd.DataFrame(means_all, columns=["Metric", "RGD", "Mutated"])
means_eq  = pd.DataFrame(means_eq,  columns=["Metric", "RGD", "Mutated"])

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), sharey=True)
w = 0.35; x = np.arange(len(means_all))
axes[0].bar(x - w/2, means_all["RGD"], w, color=PALETTE["RGD"], label="RGD")
axes[0].bar(x + w/2, means_all["Mutated"], w, color=PALETTE["Mutated"], label="Mutated")
axes[0].set_xticks(x, means_all["Metric"]); axes[0].set_title("0–100 ns")
axes[0].set_ylabel("Surface area (Å$^2$)"); axes[0].legend(frameon=False, fontsize=8)

x2 = np.arange(len(means_eq))
axes[1].bar(x2 - w/2, means_eq["RGD"], w, color=PALETTE["RGD"], label="RGD")
axes[1].bar(x2 + w/2, means_eq["Mutated"], w, color=PALETTE["Mutated"], label="Mutated")
axes[1].set_xticks(x2, means_eq["Metric"]); axes[1].set_title("Equilibrated window (t ≥ t_eq)")
for ax in axes: ax.yaxis.set_major_locator(MaxNLocator(6))
fig.suptitle("Surface metrics (RGD vs Mutated)")
savefig("Fig5_SurfaceMetrics_RGD_vs_Mut.png", fig)

# ---- Fig6: ΔRMSF (Mutated − RGD)
fig, ax = plt.subplots(figsize=(7.6, 3.6))
if {"Mutated", "RGD"}.issubset(rmsf.columns):
    delta = to_num(rmsf["Mutated"]) - to_num(rmsf["RGD"])
    xres = np.arange(len(delta))
    ax.plot(xres, delta, color=PALETTE["Mutated"], lw=1.0)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Residue index"); ax.set_ylabel("ΔRMSF (Mutated − RGD) (Å)")
    ax.set_title("Per-residue ΔRMSF (Mutated − RGD)")
    ax.yaxis.set_major_locator(MaxNLocator(6))
    savefig("Fig6_DeltaRMSF.png", fig)

# ---- Fig7–10: P–L contact per-residue (from P_L_contact)
# Mapping observed in your file preview (Pos control / RGD / Mutated blocks).  # ![alt_image](blob:https://outlook.office.com/a550e487-5903-4a79-8240-ba9502b390f1)
POS_MAP = {"res": "Unnamed: 2", "contact": "Unnamed: 3",  "score": "Unnamed: 4"}
RGD_MAP = {"res": "RGD",        "contact": "Unnamed: 9",  "score": "Unnamed: 10"}
MUT_MAP = {"res": "Mutated",    "contact": "Unnamed: 14", "score": "Unnamed: 15"}

def extract_contact_table(df, mapping):
    sub = df[[mapping["res"], mapping["contact"], mapping["score"]]].copy()
    sub.columns = ["Residue", "LigandContact", "Score"]
    sub["Score"] = to_num(sub["Score"])
    sub = sub[sub["LigandContact"].astype(str).str.upper() == "YES"]
    sub = sub.dropna(subset=["Residue", "Score"])
    sub = sub[sub["Residue"] != "ResName"]        # drop header row if present
    sub = sub.groupby("Residue", as_index=False)["Score"].max().sort_values("Score", ascending=False)
    return sub

pos_tab = extract_contact_table(pl_contacts, POS_MAP)
rgd_tab = extract_contact_table(pl_contacts, RGD_MAP)
mut_tab = extract_contact_table(pl_contacts, MUT_MAP)

def plot_top_contacts(sub, title, color, out_path, topn=25):
    top = sub.head(topn).copy()
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.barh(top["Residue"][::-1], top["Score"][::-1], color=color)
    ax.set_xlabel("Contact score (CA)"); ax.set_ylabel("Residue")
    ax.set_title(title); ax.xaxis.set_major_locator(MaxNLocator(6))
    savefig(out_path, fig)

plot_top_contacts(pos_tab, "Protein–ligand contacts (Pos control): top 25 residues",
                  PALETTE["Pos control"], "Fig7_PLContacts_PosCtrl.png")
plot_top_contacts(rgd_tab, "Protein–ligand contacts (RGD): top 25 residues",
                  PALETTE["RGD"], "Fig8_PLContacts_RGD.png")
plot_top_contacts(mut_tab, "Protein–ligand contacts (Mutated): top 25 residues",
                  PALETTE["Mutated"], "Fig9_PLContacts_Mutated.png")

# Combined comparative lollipop (union of top-25 residues across conditions)
union = pd.Index(pos_tab.head(25)["Residue"]).union(rgd_tab.head(25)["Residue"]).union(mut_tab.head(25)["Residue"])
combo = pd.DataFrame({"Residue": union})
for name, tab in [("Pos control", pos_tab), ("RGD", rgd_tab), ("Mutated", mut_tab)]:
    combo = combo.merge(tab[["Residue", "Score"]].rename(columns={"Score": name}), on="Residue", how="left")
combo = combo.fillna(0.0)

fig, ax = plt.subplots(figsize=(7.5, max(4.5, len(combo) * 0.2)))
y = np.arange(len(combo))
ax.hlines(y, xmin=0, xmax=combo[["Pos control", "RGD", "Mutated"]].max(axis=1), color="#e0e0e0", lw=0.5)
ax.plot(combo["Pos control"], y, "o", color=PALETTE["Pos control"], label="Pos control")
ax.plot(combo["RGD"], y, "o", color=PALETTE["RGD"], label="RGD")
ax.plot(combo["Mutated"], y, "o", color=PALETTE["Mutated"], label="Mutated")
ax.set_yticks(y, combo["Residue"])
ax.set_xlabel("Contact score (CA)"); ax.set_title("Protein–ligand contact scores (top‑residue union)")
ax.legend(loc="lower right", frameon=False, fontsize=8); ax.xaxis.set_major_locator(MaxNLocator(6))
savefig("Fig10_PLContacts_ComparativeLollipop.png", fig)

# ---- Fig11–12: Proxy contact timelines (from Lig_RMSD)
#  Fig11: Fraction bound (RMSD <= 4 Å) with 1-ns rolling average
dt = np.median(np.diff(to_num(lig["Time (ns)"]).dropna())) if lig.shape[0] > 1 else 0.1
win = max(1, int(round(1.0 / dt)))   # 1-ns window

fig, ax = plt.subplots(figsize=(7.2, 3.8))
for c in conds_lig:
    bound = (lig[c] <= 4.0).astype(float)
    frac = bound.rolling(win, min_periods=1).mean()
    ax.plot(lig["Time (ns)"], frac, label=c, lw=1.6, color=PALETTE.get(c))
ax.set_xlabel("Time (ns)"); ax.set_ylabel("Fraction bound (RMSD ≤ 4 Å)")
ax.set_title("Proxy P–L contact vs time (1‑ns rolling)")
ax.set_ylim(-0.02, 1.02); ax.yaxis.set_major_locator(MaxNLocator(6))
ax.legend(ncol=3, fontsize=8, frameon=False)
savefig("Fig11_PLContacts_ProxyFractionBound.png", fig)

#  Fig12: Contact index proxy = 1/(1+RMSD), smoothed and scaled 0–1 per condition
fig, ax = plt.subplots(figsize=(7.2, 3.8))
for c in conds_lig:
    r = lig[c]; idx = 1.0 / (1.0 + r)
    idx_s = idx.rolling(win, min_periods=1).mean()
    idx_s = (idx_s - idx_s.min()) / (idx_s.max() - idx_s.min() + 1e-9)
    ax.plot(lig["Time (ns)"], idx_s, label=c, lw=1.6, color=PALETTE.get(c))
ax.set_xlabel("Time (ns)"); ax.set_ylabel("Contact index (scaled 0–1)")
ax.set_title("Proxy P–L contact index vs time (1‑ns rolling)")
ax.set_ylim(-0.02, 1.02); ax.yaxis.set_major_locator(MaxNLocator(6))
ax.legend(ncol=3, fontsize=8, frameon=False)
savefig("Fig12_PLContacts_ProxyIndex.png", fig)

print("[OK] Generated all figures.")
