#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Excel summary + plots from trajectory_data.xlsx.

Added MD binding stability metrics:
- Ligand residence time (tau_bound)
- Maximum residence time (tau_max)
- Survival probability curve of binding events

Bound definition:
Ligand RMSD ≤ 4 Å
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

EXCEL_IN = "trajectory_data.xlsx"
SUMMARY_OUT = "retention_MD_summary.xlsx"

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def savefig(path, fig):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

# ------------------------------------------------
# Residence time functions
# ------------------------------------------------
def ligand_residence_events(rmsd_series, dt=0.1, threshold=4.0):

    bound = (rmsd_series <= threshold).astype(int)

    events = []
    length = 0

    for val in bound:
        if val == 1:
            length += 1
        elif length > 0:
            events.append(length)
            length = 0

    if length > 0:
        events.append(length)

    events = np.array(events) * dt

    tau_bound = events.mean() if len(events) else 0
    tau_max = events.max() if len(events) else 0

    return tau_bound, tau_max, events

# ------------------------------------------------
# Load workbook
# ------------------------------------------------
xl = pd.ExcelFile(EXCEL_IN, engine="openpyxl")
lig = xl.parse("Lig_RMSD")

for c in lig.columns:
    if c not in ["Frame", "Time (ns)"]:
        lig[c] = to_num(lig[c])

conds_lig = [c for c in lig.columns if c not in ["Frame", "Time (ns)"]]

dt = np.median(np.diff(to_num(lig["Time (ns)"]).dropna()))

# ------------------------------------------------
# Residence time analysis
# ------------------------------------------------
tau_rows = []
event_dict = {}

for c in conds_lig:

    tau_bound, tau_max, events = ligand_residence_events(lig[c], dt=dt)

    tau_rows.append({
        "Condition": c,
        "tau_bound (ns)": tau_bound,
        "tau_max (ns)": tau_max
    })

    event_dict[c] = events

tau_tbl = pd.DataFrame(tau_rows)

# ------------------------------------------------
# Write residence times to Excel
# ------------------------------------------------
if os.path.exists(SUMMARY_OUT):
    with pd.ExcelWriter(SUMMARY_OUT, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        tau_tbl.to_excel(writer, index=False, sheet_name="Ligand_residence_time")
else:
    with pd.ExcelWriter(SUMMARY_OUT, engine="openpyxl") as writer:
        tau_tbl.to_excel(writer, index=False, sheet_name="Ligand_residence_time")

print("Residence time metrics written to Excel.")

# ------------------------------------------------
# Figure: Residence time comparison
# ------------------------------------------------
fig, ax = plt.subplots(figsize=(6,4))

x = np.arange(len(tau_tbl))

ax.bar(x - 0.2, tau_tbl["tau_bound (ns)"], width=0.4, label="τ_bound")
ax.bar(x + 0.2, tau_tbl["tau_max (ns)"], width=0.4, label="τ_max")

ax.set_xticks(x)
ax.set_xticklabels(tau_tbl["Condition"], rotation=45)

ax.set_ylabel("Residence time (ns)")
ax.set_title("Ligand residence time comparison")

ax.legend()
ax.yaxis.set_major_locator(MaxNLocator(6))

savefig("Fig13_LigandResidenceTime.png", fig)

# ------------------------------------------------
# Survival curve analysis
# ------------------------------------------------
fig, ax = plt.subplots(figsize=(6,4))

for c in conds_lig:

    events = event_dict[c]

    if len(events) == 0:
        continue

    events = np.sort(events)

    survival = 1.0 - np.arange(len(events)) / len(events)

    ax.step(events, survival, where="post", label=c)

ax.set_xlabel("Residence time (ns)")
ax.set_ylabel("Survival probability  P(τ ≥ t)")
ax.set_title("Ligand binding survival curves")

ax.set_ylim(0,1.05)

ax.legend()
ax.yaxis.set_major_locator(MaxNLocator(6))

savefig("Fig14_ResidenceTimeSurvival.png", fig)

print("Generated survival curve figure.")
