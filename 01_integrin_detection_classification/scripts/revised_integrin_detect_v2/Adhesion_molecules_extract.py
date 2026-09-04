#!/usr/bin/env python3
"""
Adhesion_molecules_extract.py (UPGRADED)

Extracts and classifies adhesion-related proteins from HMMER domtblout
results into:
  - Alpha integrins
  - Beta integrins
  - Fibronectin-containing proteins

Author: Evans Asamoah Adu (extended with Copilot assistance)
"""

import os
from collections import defaultdict
from Bio import SeqIO
import pandas as pd

# ===================== USER INPUTS =====================
DOMTBL = "input/domtblout.txt"
FASTA  = "input/proteome.fa"
OUT_XLSX = "results/adhesion_molecules.xlsx"
EVALUE_CUTOFF = 1e-5

# ===================== DOMAIN DEFINITIONS =====================
ALPHA_INTEGRIN_PFAMS = {
    "PF20806", "PF20805", "PF01839", "PF13517"
}

BETA_INTEGRIN_PFAMS = {
    "PF00362", "PF08725"
}

FIBRONECTIN_PFAMS = {
    "PF00041", "PF00040", "PF26586", "PF02986"
}

ALL_DOMAINS = (
    ALPHA_INTEGRIN_PFAMS |
    BETA_INTEGRIN_PFAMS |
    FIBRONECTIN_PFAMS
)

# ===================== LOAD FASTA =====================
seqs = SeqIO.to_dict(SeqIO.parse(FASTA, "fasta"))


# ===================== PARSE domtblout =====================

hits = defaultdict(list)

with open(DOMTBL) as fh:
    for line in fh:
        if line.startswith("#"):
            continue

        cols = line.rstrip().split()
        if len(cols) < 22:
            continue

        # IMPORTANT: Pfam accession is column 2 (index 1)
        pfam_acc = cols[1].split(".")[0]

        # Query protein ID
        prot_id = cols[3]

        # Full-sequence E-value
        full_evalue = float(cols[6])

        if full_evalue <= EVALUE_CUTOFF and pfam_acc in ALL_DOMAINS:
            hits[prot_id].append((pfam_acc, full_evalue))

if not hits:
    print("WARNING: No alpha/beta integrin or fibronectin domains detected.")

# ===================== CLASSIFICATION =====================

classification = {}

for pid, domains in hits.items():
    pfams = {pf for pf, _ in domains}

    has_alpha = bool(ALPHA_INTEGRIN_PFAMS & pfams)
    has_beta = BETA_INTEGRIN_PFAMS.issubset(pfams)
    has_fn = bool(FIBRONECTIN_PFAMS & pfams)

    if has_beta:
        classification[pid] = "Beta_integrin"
    elif has_alpha:
        classification[pid] = "Alpha_integrin"
    elif has_fn:
        classification[pid] = "Fibronectin"
    else:
        classification[pid] = "Other"

# ===================== BUILD RESULT TABLE =====================

rows = []

for pid, domains in hits.items():
    for pf, ev in domains:
        rows.append({
            "Protein_ID": pid,
            "Class": classification.get(pid, "Other"),
            "Pfam": pf,
            "Evalue": ev,
            "Length": len(seqs.get(pid, "")),
        })

columns = ["Protein_ID", "Class", "Pfam", "Evalue", "Length"]
df = pd.DataFrame(rows, columns=columns)

# ===================== EXCEL OUTPUT (SAFE) =====================

os.makedirs(os.path.dirname(OUT_XLSX), exist_ok=True)

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    if df.empty:
        pd.DataFrame(
            {"Message": ["No target domains passed the E-value cutoff."]}
        ).to_excel(writer, sheet_name="NO_HITS", index=False)
    else:
        for cls in sorted(df["Class"].unique()):
            df[df["Class"] == cls].to_excel(
                writer, sheet_name=cls, index=False
            )
        df.to_excel(writer, sheet_name="ALL", index=False)

# ===================== FASTA OUTPUT =====================

FASTA_OUT = {
    "Alpha_integrin": "results/fastas/alpha_integrins.fasta",
    "Beta_integrin": "results/fastas/beta_integrins.fasta",
    "Fibronectin": "results/fastas/fibronectin.fasta",
}

for cls, out_fa in FASTA_OUT.items():
    os.makedirs(os.path.dirname(out_fa), exist_ok=True)
    with open(out_fa, "w") as handle:
        for pid, assigned in classification.items():
            if assigned == cls and pid in seqs:
                SeqIO.write(seqs[pid], handle, "fasta")

print("✅ Adhesion molecule extraction finished successfully.")
