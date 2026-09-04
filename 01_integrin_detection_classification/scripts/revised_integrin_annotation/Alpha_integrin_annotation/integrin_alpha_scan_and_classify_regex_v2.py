#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrin α motif scanning + subgroup classification (merged pipeline)
Robust edition (2026-03-07)
---------------------------------
- **α-tail variants**: strict (GFFKR/GFFRS/GFFRR), relaxed (GFF[KR][RK]), and expanded (GFF[KRHATDNQS][KRH])
- **Tail boundary**: upstream buffer and last-Cterm fallback
- **Ca2+ signatures**: legacy strict patterns + expanded DxDxDG family + EF-hand–like 12-mer window scan
- **Classification policy (default)**: derive subgroup *purely from features* in the submitted sequences
    - Non-I (parasite HENLA–DIDGDGID subclade)
    - Non-I (RGD-binding–like)
    - Non-I (laminin-binding–like, mucin-rich)
    - Unassigned non-I (from features)

Outputs CSVs with per-hit/per-sequence detail and several PNG summary panels.
"""
import argparse
import os
import re
import csv
from collections import defaultdict, Counter
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# I/O helpers (FASTA / NEXUS)
# -----------------------------
def read_fasta(path: str) -> Dict[str, str]:
    seqs: Dict[str, str] = {}
    name = None
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                name = line[1:].strip().split()[0]
                if name in seqs:
                    raise ValueError(f"Duplicate FASTA ID: {name}")
                seqs[name] = ""
            else:
                if name is None:
                    raise ValueError("FASTA format error: sequence line before first header")
                seqs[name] += line
    return seqs

def read_nexus_matrix(path: str) -> Dict[str, str]:
    seqs: Dict[str, str] = defaultdict(str)
    inside = False
    with open(path, "r") as fh:
        for line in fh:
            line = line.rstrip("\n")
            up = line.strip().upper()
            if not inside:
                if up.startswith("MATRIX"):
                    inside = True
                continue
            if line.strip().startswith(";"):
                break
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            sid = parts[0]
            chunk = "".join(parts[1:])
            seqs[sid] += chunk
    return dict(seqs)

def read_msa_auto(path: str) -> Dict[str, str]:
    # Peek header to decide parser
    with open(path, "r") as fh:
        head = "".join([next(fh, "") for _ in range(5)])
    if "#NEXUS" in head.upper():
        return read_nexus_matrix(path)
    return read_fasta(path)

# -----------------------------
# Basic utilities / heuristics
# -----------------------------
HYDRO_TM = set("AILMVFWY")
HYDRO_TAIL = set("AILVFWYMGTC")

AA = list("ACDEFGHIKLMNPQRSTVWY")
AA_COLOR = {
    "D": "#1f77b4", "E": "#1f77b4",
    "K": "#ff7f0e", "R": "#ff7f0e", "H": "#ff7f0e",
    "S": "#2ca02c", "T": "#2ca02c", "Y": "#2ca02c", "N": "#2ca02c", "Q": "#2ca02c",
    "A": "#8c564b", "V": "#8c564b", "I": "#8c564b", "L": "#8c564b", "M": "#8c564b",
    "F": "#8c564b", "W": "#8c564b", "P": "#8c564b", "G": "#8c564b", "C": "#8c564b",
}

def ungap_upper(d: Dict[str, str]) -> Dict[str, str]:
    return {k: v.replace("-", "").upper() for k, v in d.items()}

def has_signal_peptide(seq: str, n_first: int = 30, min_run: int = 7) -> bool:
    n = seq[:n_first]
    run = maxrun = 0
    for c in n:
        if c in HYDRO_TM:
            run += 1
            if run > maxrun:
                maxrun = run
        else:
            run = 0
    return maxrun >= min_run

def tm_helices(seq: str, window: int = 19, threshold: float = 0.6) -> List[Tuple[int, int]]:
    hits: List[Tuple[int, int]] = []
    L = len(seq)
    for i in range(L - window + 1):
        win = seq[i:i+window]
        prop = sum(1 for aa in win if aa in HYDRO_TM) / window
        if prop >= threshold:
            hits.append((i, i + window))
    merged: List[List[int]] = []
    for s, e in hits:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [(s, e) for s, e in merged]

def tail_after_tm(seq: str) -> Tuple[int, str]:
    best = -1
    for i in range(len(seq) - 18):
        window = seq[i:i+19]
        if sum(1 for c in window if c in HYDRO_TAIL) >= 17:
            best = i
    if best != -1:
        start = best + 19
        return start + 0, seq[start:]
    start = max(0, len(seq) - 200)
    return start + 1, seq[start:]

FG_WINDOW = 30

def count_fggap(seq: str) -> int:
    cnt = 0
    fg_pos = [m.start() for m in re.finditer("FG", seq)]
    gap_pos = [m.start() for m in re.finditer("GAP", seq)]
    j = 0
    for i in fg_pos:
        while j < len(gap_pos) and gap_pos[j] < i:
            j += 1
        k = j
        while k < len(gap_pos) and gap_pos[k] - i <= FG_WINDOW:
            cnt += 1
            k += 1
    return cnt

# -----------------------------
# Motif definitions
# -----------------------------
MOTIFS_SIMPLE = {
    'NXS/T': re.compile(r'N[^P][ST]'),
    'DXSXS': re.compile(r'D.S.S'),
    'DxD':   re.compile(r'D.D'),
    'HENLA': re.compile(r'HENLA'),
    'DIDGDGID': re.compile(r'DIDGDGID'),
    'RGD': re.compile(r'RGD'),
    'C..C': re.compile(r'C..C'),
    'GFFKR_exact': re.compile(r'GFFKR'),
    'GFFxR': re.compile(r'GFF.R'),
}

# α-tail patterns
STRICT_TAIL_PAT   = re.compile(r"(GFFKR|GFFRS|GFFRR)")
RELAXED_TAIL_PAT  = re.compile(r"GFF[KR][RK]")
EXPANDED_TAIL_PAT = re.compile(r"GFF[KRHATDNQS][KRH]")

# -----------------------------
# Ca2+ motif families and EF-hand heuristic
# -----------------------------
CA_STRICT = re.compile(r"D[DN]..DG")
CA_ALT1   = re.compile(r"D.DGDG")
CA_ALT2   = re.compile(r"D.DG[DN]G")
CA_ALT3   = re.compile(r"D[DN].D.G")
CA_FAMILY = [CA_STRICT, CA_ALT1, CA_ALT2, CA_ALT3]

def scan_ca_family(seq: str):
    hits = []
    for pat in CA_FAMILY:
        for m in pat.finditer(seq):
            hits.append(("Ca_DxDxDG_expanded", m.group(), m.start()+1, 0.0))
    return hits

def efhand_score_12mer(win: str, require_gly: bool, ban_pro: bool):
    if len(win) != 12:
        return -1.0
    if ban_pro and ('P' in win):
        return -1.0
    score = 0.0
    for i in (0,2,4):
        if win[i] in 'DE':
            score += 1.0
    if win[11] in 'DE':
        score += 1.0
    for i in (6,8):
        if win[i] in 'DE':
            score += 0.5
    has_gly = (win[5]=='G') or (win[4]=='G') or (win[6]=='G')
    if require_gly and not has_gly:
        return -1.0
    if has_gly:
        score += 0.5
    return score

def scan_efhand_12mer(seq: str, min_score: float, require_gly: bool, ban_pro: bool):
    hits = []
    L = len(seq)
    for i in range(0, max(0, L-11)):
        win = seq[i:i+12]
        s = efhand_score_12mer(win, require_gly, ban_pro)
        if s >= min_score:
            hits.append(("Ca_EFhand_like_12mer", win, i+1, s))
    return hits

# -----------------------------
# Regex-compatible motif library
# -----------------------------
# This library was derived from the motif table supplied with the phylogenetic tree.
# The patterns intentionally prioritize schistosome α-integrin subtyping:
#   α2-like: DIDGDGID-bearing β-propeller motif + GFFTR tail presentation
#   α4-like: DYDGDGNDD/FDSDGNDD β-propeller motif + shifted NGNG/SGNG-GAP-like block + GFFRR tail
#   α1/α3-like: RGD-compatible clades resolved by β-propeller and tail motifs
DEFAULT_REGEX_MOTIF_LIBRARY: List[Dict[str, object]] = [
    # Schistosome α2-like, divergent parasite-specific clade
    {"motif_id": "SCHISTO_A2_BP1_DIDGDGID", "subgroup": "Schistosome α2-like", "region": "β-propeller FG-GAP-like blade", "regex": r"F{1,2}GYSIARLGDIDGDGIDDVAISAP", "weight": 4.0, "required": True, "note": "α2-like diagnostic DIDGDGID-containing β-propeller motif"},
    {"motif_id": "SCHISTO_A2_TM_TAIL_GFFTR", "subgroup": "Schistosome α2-like", "region": "TM/cytoplasmic tail", "regex": r"VGLWILMIL[SG]ALLYC[CS]GFFTR", "weight": 4.0, "required": True, "note": "α2-like divergent GFFTR tail presentation"},
    {"motif_id": "SCHISTO_A2_NTERM_CDPLWHT", "subgroup": "Schistosome α2-like", "region": "N-terminal ectodomain", "regex": r"ISGAVIVYCDPLWHT", "weight": 1.5, "required": False, "note": "α2-like schistosome-enriched N-terminal motif"},
    {"motif_id": "SCHISTO_A2_DADGDG_WPE", "subgroup": "Schistosome α2-like", "region": "β-propeller Ca2+/FG-GAP-associated block", "regex": r"DADGDGWPEFAVTSL", "weight": 2.0, "required": False, "note": "α2-like DADGDG-WPE motif"},
    {"motif_id": "SCHISTO_A2_GWLW_ETFFKLKSD", "subgroup": "Schistosome α2-like", "region": "C-terminal ectodomain", "regex": r"GWLW[AS]ETFFKL[HK]KSD", "weight": 1.5, "required": False, "note": "α2-like membrane-proximal ectodomain motif"},
    {"motif_id": "SCHISTO_A2_KC_KQ_PLIDWSEC", "subgroup": "Schistosome α2-like", "region": "ectodomain", "regex": r"KC[NY]KQ(?:EH|GQP)L[IV]DWSEC", "weight": 1.5, "required": False, "note": "α2-like cysteine/IDWSEC motif"},
    {"motif_id": "SCHISTO_A2_FINPLE_MDFKFI", "subgroup": "Schistosome α2-like", "region": "C-terminal ectodomain", "regex": r"FINPLELTLMDFKFI", "weight": 1.5, "required": False, "note": "α2-like FINPLE-MDFKFI motif"},
    {"motif_id": "SCHISTO_A2_GPTKSTGL", "subgroup": "Schistosome α2-like", "region": "ectodomain", "regex": r"GPTKSTGL[SHI]ILTRF[HY]", "weight": 1.0, "required": False, "note": "α2-like GPTKSTGL motif"},

    # Schistosome α4-like, divergent parasite-specific clade
    {"motif_id": "SCHISTO_A4_BP1_DYDGDNDD", "subgroup": "Schistosome α4-like", "region": "β-propeller FG-GAP-like blade", "regex": r"[NH]FGF[AS]LTNLGD(?:YDGDG|FDSDG)NDD[IV][AG]VG[AS]P", "weight": 4.0, "required": True, "note": "α4-like diagnostic DYDGDGNDD/FDSDGNDD β-propeller presentation"},
    {"motif_id": "SCHISTO_A4_BP3_NGNG_GAPSHIFT", "subgroup": "Schistosome α4-like", "region": "β-propeller shifted GAP-like block", "regex": r"[NS]GNG[MI]S[VI]LRV[HN][LI]KG[PS][YF][IL]KSV[IV][IM]GAP", "weight": 3.0, "required": False, "note": "α4-like shifted NGNG/SGNG motif replacing a canonical FG-GAP-like block"},
    {"motif_id": "SCHISTO_A4_TM_TAIL_GFFRR", "subgroup": "Schistosome α4-like", "region": "TM/cytoplasmic tail", "regex": r"LGV[IL][CF]LYLLIILLYLLGFFRR", "weight": 4.0, "required": True, "note": "α4-like GFFRR tail presentation"},
    {"motif_id": "SCHISTO_A4_GWLWART_LFAKHISD", "subgroup": "Schistosome α4-like", "region": "C-terminal ectodomain", "regex": r"GWLWARTLFAKHISD", "weight": 1.5, "required": False, "note": "α4-like C-terminal ectodomain motif"},
    {"motif_id": "SCHISTO_A4_QDLS_CDPLWRA", "subgroup": "Schistosome α4-like", "region": "N-terminal ectodomain", "regex": r"QDLS(?:VFIY|IFVY|LFLY)CDPLWRA", "weight": 1.5, "required": False, "note": "α4-like CDPLWRA motif"},
    {"motif_id": "SCHISTO_A4_WLKRP_IDWSKC", "subgroup": "Schistosome α4-like", "region": "ectodomain", "regex": r"[DN]G[GS]WLKRP[FLY]IDWSKC", "weight": 1.5, "required": False, "note": "α4-like WLKRP-IDWSKC motif"},
    {"motif_id": "SCHISTO_A4_VVNPRE_IPMDV", "subgroup": "Schistosome α4-like", "region": "C-terminal ectodomain", "regex": r"VVNPRE[FL]IPMDV[KN]T[IV]", "weight": 1.5, "required": False, "note": "α4-like VVNPRE-IPMDV motif"},
    {"motif_id": "SCHISTO_A4_GPTK_MELQF", "subgroup": "Schistosome α4-like", "region": "ectodomain", "regex": r"GPT[QK][ASQ][QK]G[TI]RMELQF[HY]", "weight": 1.0, "required": False, "note": "α4-like GPTK/GPTQ-MELQF motif"},
    {"motif_id": "SCHISTO_A4_WIADCH_FPPW", "subgroup": "Schistosome α4-like", "region": "ectodomain", "regex": r"R[DY][QKT]WIADCH[YF][IL]FPPW", "weight": 2.0, "required": False, "note": "α4-like WIADCH-FPPW motif"},

    # Schistosome α1-like / RGD-compatible clade
    {"motif_id": "SCHISTO_A1_BP1_RFGHALTN", "subgroup": "Schistosome α1-like", "region": "β-propeller FG-GAP-like blade", "regex": r"RFGHALTNIGD[IMV]DGDGTEDLAVSCP", "weight": 4.0, "required": True, "note": "α1-like RFGHALTN β-propeller motif"},
    {"motif_id": "SCHISTO_A1_BP2_SYFGYS", "subgroup": "Schistosome α1-like", "region": "β-propeller FG-GAP-like blade", "regex": r"SYFGYSLAVADLDGNGL[AV]DIIVGAP", "weight": 2.5, "required": False, "note": "α1-like second FG-GAP-like motif"},
    {"motif_id": "SCHISTO_A1_TM_TAIL_GFFHR", "subgroup": "Schistosome α1-like", "region": "TM/cytoplasmic tail", "regex": r"LGLALLALLIFTMWRCGFFHR|GFFHR", "weight": 3.0, "required": False, "note": "α1-like GFFHR tail variant"},
    {"motif_id": "SCHISTO_A1_DLDGNHAP", "subgroup": "Schistosome α1-like", "region": "β-propeller Ca2+/FG-GAP-associated block", "regex": r"DLDGN[HY]APDLV[IV]GDY", "weight": 1.5, "required": False, "note": "α1-like DLDGNH/YAP motif"},
    {"motif_id": "SCHISTO_A1_GWIWAD", "subgroup": "Schistosome α1-like", "region": "C-terminal ectodomain", "regex": r"GWIWADTFFR[HY]KISD", "weight": 1.5, "required": False, "note": "α1-like GWIWAD motif"},

    # Schistosome α3-like / RGD-compatible clade
    {"motif_id": "SCHISTO_A3_BP1_GFGATITK", "subgroup": "Schistosome α3-like", "region": "β-propeller FG-GAP-like blade", "regex": r"GFGA[AT]ITKLGDINHDG[YF]QD[FL][AI]?[VI]GAP", "weight": 4.0, "required": True, "note": "α3-like GFGAT/AITK β-propeller motif"},
    {"motif_id": "SCHISTO_A3_BP2_SRFGHSL", "subgroup": "Schistosome α3-like", "region": "β-propeller FG-GAP-like blade", "regex": r"SRFGHSL[LI]F[LI]DINGD[NG]WDDLI[IV]GAP", "weight": 2.5, "required": False, "note": "α3-like second FG-GAP-like motif"},
    {"motif_id": "SCHISTO_A3_TM_TAIL_GFFKR", "subgroup": "Schistosome α3-like", "region": "TM/cytoplasmic tail", "regex": r"GGLLLLSILVIILYKAGFFKR|GFFKR", "weight": 2.0, "required": False, "note": "α3-like canonical GFFKR tail"},
    {"motif_id": "SCHISTO_A3_DLDENGYP", "subgroup": "Schistosome α3-like", "region": "β-propeller Ca2+/FG-GAP-associated block", "regex": r"DLDENGYPDMAIGA[AT]", "weight": 1.5, "required": False, "note": "α3-like DLDENGYP motif"},
    {"motif_id": "SCHISTO_A3_SSVAVLRARPVVK", "subgroup": "Schistosome α3-like", "region": "ectodomain", "regex": r"SSVAVLRARPVVKL[RN]", "weight": 1.5, "required": False, "note": "α3-like SSVAVLRARPVVK motif"},

    # General α-integrin features retained as annotation-level evidence, not subtype-defining.
    {"motif_id": "ALPHA_GENERAL_RGD", "subgroup": "General α-integrin annotation", "region": "ligand-binding ectodomain", "regex": r"RGD", "weight": 0.5, "required": False, "note": "RGD-compatible motif; annotative only"},
    {"motif_id": "ALPHA_GENERAL_NGLYCO", "subgroup": "General α-integrin annotation", "region": "ectodomain", "regex": r"N[^P][ST]", "weight": 0.0, "required": False, "note": "N-glycosylation sequon; density reported separately"},
    {"motif_id": "ALPHA_GENERAL_GFFXR_TAIL", "subgroup": "General α-integrin annotation", "region": "cytoplasmic tail", "regex": r"GFF.R", "weight": 0.5, "required": False, "note": "Generic α-integrin GFFxR tail family"},
]

def export_default_regex_motif_library(path: str) -> None:
    """Write the built-in regex motif library as TSV for editing/versioning."""
    cols = ["motif_id", "subgroup", "region", "regex", "weight", "required", "note"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        writer.writeheader()
        for row in DEFAULT_REGEX_MOTIF_LIBRARY:
            writer.writerow({c: row.get(c, "") for c in cols})

def load_regex_motif_library(path: Optional[str]) -> List[Dict[str, object]]:
    """Load a user-supplied motif library TSV/CSV or use the built-in library.
    Required columns: motif_id, subgroup, regex. Optional: region, weight, required, note.
    """
    if not path:
        return DEFAULT_REGEX_MOTIF_LIBRARY
    sep = "\t" if path.lower().endswith((".tsv", ".tab")) else ","
    df = pd.read_csv(path, sep=sep)
    required_cols = {"motif_id", "subgroup", "regex"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Motif library missing required columns: {missing}")
    rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        rec = row.to_dict()
        rec["weight"] = float(rec.get("weight", 1.0) if not pd.isna(rec.get("weight", 1.0)) else 1.0)
        req = rec.get("required", False)
        if isinstance(req, str):
            rec["required"] = req.strip().lower() in {"1", "true", "yes", "y"}
        else:
            rec["required"] = bool(req)
        rows.append(rec)
    return rows

def scan_regex_motif_library(seq: str, library: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Return all regex motif hits in a sequence."""
    hits: List[Dict[str, object]] = []
    for rec in library:
        pattern = str(rec["regex"])
        try:
            pat = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"Invalid regex for motif_id={rec.get('motif_id')}: {pattern} ({exc})")
        for m in pat.finditer(seq):
            hits.append({
                "motif_id": rec.get("motif_id", ""),
                "subgroup": rec.get("subgroup", ""),
                "region": rec.get("region", ""),
                "regex": pattern,
                "match": m.group(),
                "abs_pos_1based": m.start() + 1,
                "weight": float(rec.get("weight", 1.0)),
                "required": bool(rec.get("required", False)),
                "note": rec.get("note", ""),
            })
    return hits

def score_regex_hits(hits: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    """Collapse motif hits into subgroup scores using unique motif IDs."""
    scores: Dict[str, Dict[str, object]] = defaultdict(lambda: {"score": 0.0, "motif_ids": [], "required_hit": False, "hit_count": 0})
    seen = set()
    for h in hits:
        subgroup = str(h.get("subgroup", ""))
        if subgroup == "General α-integrin annotation":
            continue
        motif_id = str(h.get("motif_id", ""))
        key = (subgroup, motif_id)
        scores[subgroup]["hit_count"] += 1
        if key in seen:
            continue
        seen.add(key)
        scores[subgroup]["score"] += float(h.get("weight", 1.0))
        scores[subgroup]["motif_ids"].append(motif_id)
        if bool(h.get("required", False)):
            scores[subgroup]["required_hit"] = True
    return dict(scores)

def format_regex_scores(scores: Dict[str, Dict[str, object]]) -> str:
    parts = []
    for subgroup, info in sorted(scores.items(), key=lambda kv: (-float(kv[1]["score"]), kv[0])):
        parts.append(f"{subgroup}:{float(info['score']):.1f}")
    return ";".join(parts)

def classify_by_regex_library(metrics: dict) -> Tuple[str, str]:
    """Subtype classifier using the regex library and legacy features.

    Critical rule: α2-like and α4-like are separated by their specific motif presentation,
    not by a generic parasite-HENLA/DIDGDGID rule. α2-like is driven by the
    DIDGDGID + GFFTR presentation; α4-like is driven by the DYDGDGNDD/FDSDGNDD,
    shifted NGNG/SGNG-GAP-like block, and GFFRR presentation.
    """
    scores: Dict[str, Dict[str, object]] = metrics.get("regex_scores", {}) or {}
    hits: List[Dict[str, object]] = metrics.get("regex_hits", []) or []
    rgd = bool(metrics.get("RGD_flag", False))
    has_tail = bool(metrics.get("has_alpha_tail", False))
    L = max(1, int(metrics.get("Length", 0)))
    nxst_per100 = 100.0 * float(metrics.get("NXS/T_count", 0)) / L

    def has_motif(motif_id: str) -> bool:
        return any(str(h.get("motif_id")) == motif_id for h in hits)

    def subgroup_score(name: str) -> float:
        return float(scores.get(name, {}).get("score", 0.0))

    def subgroup_motifs(name: str) -> List[str]:
        return list(scores.get(name, {}).get("motif_ids", []))

    # Strong parasite divergent α2/α4 calls first.
    a2_score = subgroup_score("Schistosome α2-like")
    a4_score = subgroup_score("Schistosome α4-like")
    a1_score = subgroup_score("Schistosome α1-like")
    a3_score = subgroup_score("Schistosome α3-like")

    a2_required = has_motif("SCHISTO_A2_BP1_DIDGDGID") or has_motif("SCHISTO_A2_TM_TAIL_GFFTR")
    a4_required = has_motif("SCHISTO_A4_BP1_DYDGDNDD") or has_motif("SCHISTO_A4_TM_TAIL_GFFRR") or has_motif("SCHISTO_A4_BP3_NGNG_GAPSHIFT")

    if a2_score >= 6.0 and a2_required and a2_score >= a4_score + 2.0:
        return "Schistosome α2-like divergent α-integrin", f"regex score {a2_score:.1f}; motifs={','.join(subgroup_motifs('Schistosome α2-like'))}"
    if a4_score >= 6.0 and a4_required and a4_score >= a2_score + 2.0:
        return "Schistosome α4-like divergent α-integrin", f"regex score {a4_score:.1f}; motifs={','.join(subgroup_motifs('Schistosome α4-like'))}"

    # RGD-compatible α1/α3-like candidates, separated by clade-specific β-propeller motifs.
    if a1_score >= 5.0 and has_motif("SCHISTO_A1_BP1_RFGHALTN") and a1_score >= a3_score + 1.0:
        suffix = "; RGD-compatible" if rgd else ""
        return "Schistosome α1-like candidate α-integrin", f"regex score {a1_score:.1f}; motifs={','.join(subgroup_motifs('Schistosome α1-like'))}{suffix}"
    if a3_score >= 5.0 and has_motif("SCHISTO_A3_BP1_GFGATITK") and a3_score >= a1_score + 1.0:
        suffix = "; RGD-compatible" if rgd else ""
        return "Schistosome α3-like candidate α-integrin", f"regex score {a3_score:.1f}; motifs={','.join(subgroup_motifs('Schistosome α3-like'))}{suffix}"

    # Ambiguous α1/α3-like.
    if max(a1_score, a3_score) >= 4.0:
        best = "Schistosome α1-like" if a1_score >= a3_score else "Schistosome α3-like"
        return "Schistosome α1/α3-like candidate, subtype unresolved", f"best regex={best}; scores={format_regex_scores(scores)}"

    # Legacy feature-level labels retained as lower-confidence fallbacks.
    if rgd and has_tail:
        return "Schistosome RGD-compatible α-integrin candidate", "RGD present & α-tail hallmark/variant present; no subtype-specific regex threshold reached"
    if (not rgd) and nxst_per100 >= 1.0 and has_tail:
        return "Schistosome non-I α-integrin candidate, mucin/glycan-rich", f"NXS/T density {nxst_per100:.2f} per 100 aa, α-tail present, no subtype-specific regex threshold reached"
    if has_tail or int(metrics.get("FGGAP_heuristic_count", 0)) >= 2:
        return "Schistosome α-integrin-like, unassigned subtype", "general α-integrin features present but subtype-specific motif evidence is insufficient"
    return "Unassigned", "No α-integrin subtype rule matched"


# -----------------------------
# Group mapping (optional)
# -----------------------------

def load_groups(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    m: Dict[str, str] = {}
    with open(path, "r") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            sid, grp = parts[0], parts[1]
            m[sid] = grp
    return m

def load_subgroup_mapping(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    if path.lower().endswith('.tsv'):
        df = pd.read_csv(path, sep='\t')
    else:
        df = pd.read_csv(path)
    needed = {'ID', 'Alpha_integrin_subgroup'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Mapping file missing required columns: {missing}")
    return dict(zip(df['ID'], df['Alpha_integrin_subgroup']))

# -----------------------------
# Feature-derived classifier (default)
# -----------------------------

def classify_from_features(metrics: dict) -> Tuple[str, str]:
    """Return (label, reason) using regex subtype library plus legacy feature fallback."""
    return classify_by_regex_library(metrics)

# -----------------------------
# Tail scanner (strict, relaxed, expanded)
# -----------------------------

def scan_alpha_tail(region_seq_u: str, offset_1based: int, use_relaxed: bool = True, use_expanded: bool = True):
    hits_strict: List[Tuple[str, int, int]] = []
    spans_strict: List[Tuple[int, int]] = []
    for m in STRICT_TAIL_PAT.finditer(region_seq_u):
        motif = m.group()
        abspos = offset_1based + m.start()
        after = region_seq_u[m.end():m.end()+10]
        kr_ct = sum(1 for c in after if c in "KR")
        hits_strict.append((motif, abspos, kr_ct))
        spans_strict.append((m.start(), m.end()))

    hits_relaxed: List[Tuple[str, int, int]] = []
    if use_relaxed:
        for m in RELAXED_TAIL_PAT.finditer(region_seq_u):
            start, end = m.start(), m.end()
            if any(not (end <= s or start >= e) for (s, e) in spans_strict):
                continue
            motif = m.group()
            abspos = offset_1based + start
            after = region_seq_u[end:end+10]
            kr_ct = sum(1 for c in after if c in "KR")
            hits_relaxed.append((motif, abspos, kr_ct))

    hits_expanded: List[Tuple[str, int, int]] = []
    if use_expanded:
        relaxed_spans = [(m.start(), m.end()) for m in RELAXED_TAIL_PAT.finditer(region_seq_u)] if use_relaxed else []
        for m in EXPANDED_TAIL_PAT.finditer(region_seq_u):
            start, end = m.start(), m.end()
            if any(not (end <= s or start >= e) for (s, e) in spans_strict):
                continue
            if any(not (end <= s or start >= e) for (s, e) in relaxed_spans):
                continue
            motif = m.group()
            abspos = offset_1based + start
            after = region_seq_u[end:end+10]
            kr_ct = sum(1 for c in after if c in "KRH")
            hits_expanded.append((motif, abspos, kr_ct))

    return hits_strict, hits_relaxed, hits_expanded

# -----------------------------
# Core analysis
# -----------------------------

def analyze(in_path: str, outdir: str, groups_path: Optional[str], user_map_path: Optional[str],
            use_relaxed_alpha: bool, write_excel: bool,
            tail_buffer: int = 12, tail_fallback_len: int = 80,
            classify_mode: str = 'features',
            ca_scan_mode: str = 'both', ef_min_score: float = 3.5, ef_require_gly: bool = False, ef_ban_proline: bool = False,
            motif_library_path: Optional[str] = None, export_motif_library: Optional[str] = None):
    os.makedirs(outdir, exist_ok=True)
    aligned = read_msa_auto(in_path)
    raw = ungap_upper(aligned)

    gmap = load_groups(groups_path)
    groups = sorted(set(gmap.values())) if gmap else ["all"]

    submap = load_subgroup_mapping(user_map_path)
    regex_library = load_regex_motif_library(motif_library_path)
    if export_motif_library:
        export_default_regex_motif_library(export_motif_library)

    tail_rows: List[List[object]] = []
    regex_rows: List[Dict[str, object]] = []
    regex_score_rows: List[Dict[str, object]] = []
    ca_rows: List[List[object]] = []
    fggap_rows: List[List[object]] = []
    nglyco_rows: List[List[object]] = []
    lineage_rows: List[List[object]] = []

    tail_counts = {g: Counter() for g in groups}
    kr_hist = {g: [] for g in groups}
    ca_counts = {g: Counter() for g in groups}
    mid_counts = {g: Counter() for g in groups}
    mid5_all: List[str] = []

    feature_rows: List[Dict[str, object]] = []

    for sid, seq in raw.items():
        grp = gmap.get(sid, "all") if gmap else "all"
        L = len(seq)
        sig = has_signal_peptide(seq)
        tms = tm_helices(seq)
        cterm_tm = any(e > L - 40 for (s, e) in tms)

        simple_counts = {k: len(p.findall(seq)) for k, p in MOTIFS_SIMPLE.items()}
        gffxr_near = bool(MOTIFS_SIMPLE['GFFxR'].search(seq[-140:])) if L >= 10 else False

        # Phylogeny-guided regex motif library scan
        regex_hits = scan_regex_motif_library(seq, regex_library)
        regex_scores = score_regex_hits(regex_hits)
        for h in regex_hits:
            regex_rows.append({
                "seq_id": sid,
                "group": grp,
                "motif_id": h["motif_id"],
                "subgroup": h["subgroup"],
                "region": h["region"],
                "regex": h["regex"],
                "match": h["match"],
                "abs_pos_1based": h["abs_pos_1based"],
                "weight": h["weight"],
                "required": h["required"],
                "note": h["note"],
            })
        for subgroup, info in regex_scores.items():
            regex_score_rows.append({
                "seq_id": sid,
                "group": grp,
                "subgroup": subgroup,
                "regex_score": float(info["score"]),
                "motif_ids": ",".join(info["motif_ids"]),
                "required_hit": bool(info["required_hit"]),
                "hit_count": int(info["hit_count"]),
            })

        # Tail scan with upstream buffer
        tstart1, _tail = tail_after_tm(seq)
        buf = max(0, tail_buffer)
        scan_start = max(1, tstart1 - buf)
        scan_region = seq[scan_start-1:]
        hits_strict, hits_relaxed, hits_expanded = scan_alpha_tail(scan_region, scan_start, use_relaxed_alpha, True)

        # Fallback: last-N aa if nothing found
        if not (hits_strict or hits_relaxed or hits_expanded):
            lastN = seq[-tail_fallback_len:] if L >= tail_fallback_len else seq
            offs = L - len(lastN) + 1
            fb_strict, fb_relaxed, fb_expanded = scan_alpha_tail(lastN, offs, use_relaxed_alpha, True)
            if fb_strict or fb_relaxed or fb_expanded:
                hits_strict, hits_relaxed, hits_expanded = fb_strict, fb_relaxed, fb_expanded

        for motif, abspos, kr_ct in hits_strict:
            tail_rows.append([sid, grp, motif, abspos, kr_ct])
            tail_counts[grp][motif] += 1
            kr_hist[grp].append(kr_ct)
        for motif, abspos, kr_ct in hits_relaxed:
            label = "GFF[KR][RK]_relaxed"
            tail_rows.append([sid, grp, label, abspos, kr_ct])
            tail_counts[grp][label] += 1
            kr_hist[grp].append(kr_ct)
        for motif, abspos, kr_ct in hits_expanded:
            label = "GFFKR_family_expanded"
            tail_rows.append([sid, grp, f"{label}:{motif}", abspos, kr_ct])
            tail_counts[grp][label] += 1
            kr_hist[grp].append(kr_ct)

        # Ca2+ signatures
        if ca_scan_mode in ('strict','both'):
            for m in re.finditer(r"D[DN]..DG", seq):
                ca_rows.append([sid, grp, "Dx[DN]xDG", m.group(), m.start()+1])
                ca_counts[grp]["Dx[DN]xDG"] += 1
            for m in re.finditer(r"N.D...D", seq):
                ca_rows.append([sid, grp, "NxDxxxD", m.group(), m.start()+1])
                ca_counts[grp]["NxDxxxD"] += 1
        if ca_scan_mode in ('expanded','both'):
            for cls, motif, pos1, _ in scan_ca_family(seq):
                ca_rows.append([sid, grp, cls, motif, pos1])
                ca_counts[grp][cls] += 1
        if ca_scan_mode in ('efhand','both'):
            for cls, motif, pos1, score in scan_efhand_12mer(seq, ef_min_score, ef_require_gly, ef_ban_proline):
                ca_rows.append([sid, grp, f"{cls}|score={score:.2f}", motif, pos1])
                ca_counts[grp][cls] += 1

        # alphaI-like 5-mers (summary only)
        dxsxs = dxsxt = dxsxX = 0
        for i in range(L - 4):
            five = seq[i:i+5]
            if five[0] == 'D' and five[2] == 'S':
                if five[4] == 'S':
                    dxsxs += 1; mid_counts[grp]["DXSXS"] += 1
                elif five[4] == 'T':
                    dxsxt += 1; mid_counts[grp]["DXSXT"] += 1
                else:
                    dxsxX += 1; mid_counts[grp]["DXSX*"] += 1
                mid5_all.append(five)

        fgc = count_fggap(seq)
        fggap_rows.append([sid, grp, fgc])

        ng = 0
        for i in range(L - 2):
            if seq[i] == 'N' and seq[i+1] != 'P' and seq[i+2] in 'ST':
                ng += 1
        nglyco_rows.append([sid, grp, ng])

        henla = didg = 0
        for m in re.finditer("HENLA", seq):
            lineage_rows.append([sid, grp, "HENLA", m.start()+1])
            henla += 1
        for m in re.finditer("DIDGDGID", seq):
            lineage_rows.append([sid, grp, "DIDGDGID", m.start()+1])
            didg += 1

        # Subgroup by features (default), or map, or none
        if classify_mode == 'features':
            subgroup, reason = classify_from_features({
                'Length': L,
                'NXS/T_count': simple_counts['NXS/T'],
                'RGD_flag': simple_counts['RGD']>0,
                'HENLA_count': henla,
                'DIDGDGID_count': didg,
                'GFFKR_count': sum(1 for h in hits_strict if h[0]=='GFFKR'),
                'GFFRS_count': sum(1 for h in hits_strict if h[0]=='GFFRS'),
                'GFFRR_count': sum(1 for h in hits_strict if h[0]=='GFFRR'),
                'GFF_relaxed_count': len(hits_relaxed),
                'GFFKR_family_expanded_count': len(hits_expanded),
                'has_alpha_tail': bool(hits_strict or hits_relaxed or hits_expanded or gffxr_near),
                'FGGAP_heuristic_count': fgc,
                'regex_hits': regex_hits,
                'regex_scores': regex_scores,
            })
        elif classify_mode == 'map' and sid in submap:
            subgroup, reason = submap[sid], 'from --map'
        elif classify_mode == 'none':
            subgroup, reason = '', ''
        else:
            subgroup, reason = 'Unassigned non-I (from features)', 'fallback'

        feature_rows.append({
            'ID': sid,
            'group': grp,
            'Alpha_integrin_subgroup': subgroup,
            'Alpha_integrin_subgroup_reason': reason,
            'Regex_subgroup_scores': format_regex_scores(regex_scores),
            'Regex_best_subgroup': max(regex_scores.items(), key=lambda kv: float(kv[1]["score"]))[0] if regex_scores else "",
            'Regex_best_score': max((float(v["score"]) for v in regex_scores.values()), default=0.0),
            'Regex_motif_ids': ",".join([str(h["motif_id"]) for h in regex_hits if str(h.get("subgroup","")) != "General α-integrin annotation"]),
            'Length': L,
            'SignalPeptide': sig,
            'Cterm_TM': cterm_tm,
            'Tail_start_1based': tstart1,
            'GFFKR_count': sum(1 for h in hits_strict if h[0] == 'GFFKR'),
            'GFFRS_count': sum(1 for h in hits_strict if h[0] == 'GFFRS'),
            'GFFRR_count': sum(1 for h in hits_strict if h[0] == 'GFFRR'),
            'GFF_relaxed_count': len(hits_relaxed),
            'GFFKR_family_expanded_count': len(hits_expanded),
            'GFFKR_exact_flag': simple_counts['GFFKR_exact'] > 0,
            'GFFxR_near_Cterm': gffxr_near,
            'NXS/T_count': simple_counts['NXS/T'],
            'DXSXS_count': simple_counts['DXSXS'],
            'DxD_count': simple_counts['DxD'],
            'RGD_flag': simple_counts['RGD'] > 0,
            'C..C_flag': simple_counts['C..C'] > 0,
            'HENLA_count': henla,
            'DIDGDGID_count': didg,
            'Ca_Dx[DN]xDG_count': sum(1 for r in ca_rows if r[0]==sid and r[2]=="Dx[DN]xDG"),
            'Ca_NxDxxxD_count': sum(1 for r in ca_rows if r[0]==sid and r[2]=="NxDxxxD"),
            'alphaI_DXSXS_count': dxsxs,
            'alphaI_DXSXT_count': dxsxt,
            'alphaI_DXSX*_count': dxsxX,
            'FGGAP_heuristic_count': fgc,
            'NXS_T_count_per_seq': ng,
        })

    # Detailed CSVs
    pd.DataFrame(
        tail_rows,
        columns=["seq_id", "group", "alpha_tail_motif", "abs_pos_1based", "KR_count_plus10"]
    ).to_csv(os.path.join(outdir, "alpha_tail_motifs.csv"), index=False)

    pd.DataFrame(
        ca_rows, columns=["seq_id", "group", "signature_class", "motif", "abs_pos_1based"]
    ).to_csv(os.path.join(outdir, "alpha_Ca_binding_signatures.csv"), index=False)

    pd.DataFrame(
        fggap_rows, columns=["seq_id", "group", "FGGAP_heuristic_count"]
    ).to_csv(os.path.join(outdir, "alpha_FGGAP_counts.csv"), index=False)

    pd.DataFrame(
        nglyco_rows, columns=["seq_id", "group", "NXS_T_count"]
    ).to_csv(os.path.join(outdir, "alpha_N_glyco_counts.csv"), index=False)

    pd.DataFrame(
        lineage_rows, columns=["seq_id", "group", "motif", "abs_pos_1based"]
    ).to_csv(os.path.join(outdir, "alpha_lineage_motifs.csv"), index=False)

    pd.DataFrame(regex_rows).to_csv(os.path.join(outdir, "alpha_regex_motif_hits.csv"), index=False)
    pd.DataFrame(regex_score_rows).to_csv(os.path.join(outdir, "alpha_regex_subgroup_scores.csv"), index=False)

    # Summary CSV for quick paneling
    ca_per_seq = defaultdict(lambda: {"Dx[DN]xDG": 0, "NxDxxxD": 0})
    for sid, grp, cls, motif, pos in ca_rows:
        if cls in ca_per_seq[sid]:
            ca_per_seq[sid][cls] += 1

    summary_rows = []
    for sid, seq in raw.items():
        grp = next((r[1] for r in fggap_rows if r[0] == sid), (gmap.get(sid, "all") if gmap else "all"))
        gffkr = sum(1 for r in tail_rows if r[0] == sid and r[2] == "GFFKR")
        gffrs = sum(1 for r in tail_rows if r[0] == sid and r[2] == "GFFRS")
        gffrr = sum(1 for r in tail_rows if r[0] == sid and r[2] == "GFFRR")
        grel  = sum(1 for r in tail_rows if r[0] == sid and r[2] == "GFF[KR][RK]_relaxed")
        gexp  = sum(1 for r in tail_rows if r[0] == sid and r[2].startswith("GFFKR_family_expanded"))
        fgc   = next((r[2] for r in fggap_rows if r[0] == sid), 0)
        ngc   = next((r[2] for r in nglyco_rows if r[0] == sid), 0)
        henla = sum(1 for r in lineage_rows if r[0] == sid and r[2] == "HENLA")
        didg  = sum(1 for r in lineage_rows if r[0] == sid and r[2] == "DIDGDGID")
        tstart1, _ = tail_after_tm(seq)
        summary_rows.append([
            sid, grp, len(seq), tstart1,
            gffkr, gffrs, gffrr, grel, gexp,
            fgc, ngc, ca_per_seq[sid]["Dx[DN]xDG"], ca_per_seq[sid]["NxDxxxD"],
            0, 0,
            henla, didg
        ])

    summary_cols = [
        "seq_id", "group", "length", "tail_start_1based",
        "GFFKR_count", "GFFRS_count", "GFFRR_count", "GFF_relaxed_count", "GFFKR_family_expanded_count",
        "FGGAP_heuristic_count", "NXS_T_count",
        "Ca_Dx[DN]xDG_count", "Ca_NxDxxxD_count",
        "alphaI_DXSXS_count", "alphaI_DXSXT_count",
        "HENLA_count", "DIDGDGID_count",
    ]
    pd.DataFrame(summary_rows, columns=summary_cols).to_csv(
        os.path.join(outdir, "alpha_seq_summary.csv"), index=False
    )

    # Consolidated features table
    feature_df = pd.DataFrame(feature_rows)
    feat_csv = os.path.join(outdir, 'alpha_integrin_features.csv')
    feature_df.to_csv(feat_csv, index=False)
    if write_excel:
        feature_df.to_excel(os.path.join(outdir, 'alpha_integrin_features.xlsx'), index=False)

    # ------------------ Figures ------------------
    labels = groups
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=160)
    keys = ["GFFKR", "GFFRS", "GFFRR", "GFF[KR][RK]_relaxed", "GFFKR_family_expanded"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#8c564b", "#9467bd"]
    x = np.arange(len(labels)); width = 0.16
    for i, k in enumerate(keys):
        vals = [sum(1 for r in tail_rows if r[1]==g and (r[2]==k or (k=="GFFKR_family_expanded" and r[2].startswith("GFFKR_family_expanded")))) for g in labels]
        axes[0].bar(x + (i - (len(keys)-1)/2) * width, vals, width, label=k, color=colors[i])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Motif count (α tails)")
    axes[0].set_title("α-tail hallmark & expanded variants")
    axes[0].legend(frameon=False, fontsize=8)

    # Ca2+ overview + Nglyco + FG-GAP
    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4), dpi=160)
    cats = ["Dx[DN]xDG", "NxDxxxD", "Ca_DxDxDG_expanded", "Ca_EFhand_like_12mer"]
    cols = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    vals = [sum(ca_counts[g].get(c,0) for g in labels) for c in cats]
    axes2[0].bar(range(len(cats)), vals, color=cols)
    axes2[0].set_xticks(range(len(cats)))
    axes2[0].set_xticklabels(cats, rotation=10)
    axes2[0].set_ylabel("Total hits")
    axes2[0].set_title("Ca2+-binding signatures (all)")

    grouped_ng = [[r[2] for r in nglyco_rows if r[1] == g] for g in labels] if len(labels) > 1 else [[r[2] for r in nglyco_rows]]
    axes2[1].boxplot(grouped_ng, showfliers=False, labels=labels if labels else ["all"])
    axes2[1].set_ylabel("NXS/T sites per sequence")
    axes2[1].set_title("N-glycosylation site counts")

    grouped_fg = [[r[2] for r in fggap_rows if r[1] == g] for g in labels] if len(labels) > 1 else [[r[2] for r in fggap_rows]]
    axes2[2].boxplot(grouped_fg, showfliers=False, labels=labels if labels else ["all"])
    axes2[2].set_ylabel("FG–GAP heur. count")
    axes2[2].set_title("FG–GAP repeat counts (heuristic)")

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "alpha_tail_panels.png"), bbox_inches="tight")
    fig2.savefig(os.path.join(outdir, "alpha_Ca_Ngly_FGGAP_panels.png"), bbox_inches="tight")
    plt.close(fig); plt.close(fig2)

    # alphaI MIDAS 5-mer overview
    idx = {a: i for i, a in enumerate(AA)}
    M = np.zeros((20, 5), float)
    for s in mid5_all:
        if len(s) != 5:
            continue
        for j, ch in enumerate(s):
            if ch in idx:
                M[idx[ch], j] += 1
    colsum = M.sum(axis=0, keepdims=True)
    colsum[colsum == 0] = 1
    M = M / colsum

    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=160)
    total = Counter()
    for g in labels:
        total.update(mid_counts[g])
    ax1.bar([0, 1, 2], [total["DXSXS"], total["DXSXT"], total["DXSX*"]], color=["#2ca02c", "#ff7f0e", "#8c564b"])
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(["DXSXS", "DXSXT", "DXSX*"])
    ax1.set_ylabel("Count")
    ax1.set_title("alphaI MIDAS 5-mer classes (combined)")

    ax2.set_xlim(0, 5); ax2.set_ylim(0, 1)
    ax2.set_xticks(range(5)); ax2.set_xticklabels(["1", "2", "3", "4", "5"])
    ax2.set_yticks([0, 0.5, 1.0]); ax2.set_ylabel("Freq")
    ax2.set_title("alphaI MIDAS 5-mer logo (combined)")
    for j in range(5):
        col = [(AA[i], M[i, j]) for i in range(20) if M[i, j] > 0]
        col.sort(key=lambda x: x[1])
        y = 0
        for aa, h in col:
            ax2.text(j + 0.05, y, aa, fontsize=12, color=AA_COLOR.get(aa, "black"), va="bottom", ha="left", fontweight="bold")
            y += h
    ax2.set_xlim(-0.1, 5.1)

    plt.tight_layout()
    fig3.savefig(os.path.join(outdir, "alphaI_MIDAS_panels.png"), bbox_inches="tight")
    plt.close(fig3)

    # Lineage motif bar
    lin_counts = Counter([r[2] for r in lineage_rows])
    fig4, ax4 = plt.subplots(figsize=(5, 4), dpi=160)
    ax4.bar(range(2), [lin_counts.get("HENLA", 0), lin_counts.get("DIDGDGID", 0)], color=["#7f7f7f", "#bcbd22"])
    ax4.set_xticks(range(2)); ax4.set_xticklabels(["HENLA", "DIDGDGID"]) 
    ax4.set_ylabel("Total hits"); ax4.set_title("Lineage-specific motif counts")
    plt.tight_layout(); fig4.savefig(os.path.join(outdir, "alpha_lineage_motifs.png"), bbox_inches="tight"); plt.close(fig4)

    print("[OK] Wrote outputs to:", outdir)
    print(" -", os.path.join(outdir, 'alpha_integrin_features.csv'))


def main():
    ap = argparse.ArgumentParser(description="Schistosome integrin α motif scanning and feature-derived subgroup classification")
    ap.add_argument("--in", dest="in_path", required=True, help="Input FASTA or NEXUS alignment")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--groups", help="Optional 2-col TSV: seq_id\tgroup")
    ap.add_argument("--map", dest="user_map", help="Optional subgroup map CSV/TSV with columns: ID, Alpha_integrin_subgroup")
    ap.add_argument("--no-relaxed-alpha", action="store_true", help="Disable relaxed GFF[KR][RK] detection (strict-only)")
    ap.add_argument("--excel", action="store_true", help="Also write alpha_integrin_features.xlsx")
    ap.add_argument("--tail-buffer", type=int, default=12, help="Scan this many aa upstream of predicted tail start (default: 12)")
    ap.add_argument("--tail-fallback-len", type=int, default=80, help="Length of last-Cterm window to scan if tail scan finds nothing (default: 80)")
    ap.add_argument("--classify-mode", choices=["features","map","none"], default="features", help="How to assign subgroup labels: features (regex-library rules), map (from --map), or none (blank)")
    ap.add_argument("--motif-library", help="Optional regex motif library TSV/CSV with columns: motif_id, subgroup, regex, weight, required, region, note")
    ap.add_argument("--export-default-motif-library", help="Write the built-in regex motif library TSV to this path and continue analysis")
    ap.add_argument("--ca-scan-mode", choices=["strict","expanded","efhand","both"], default="both",
                   help="Alpha Ca2+ scan: strict (legacy), expanded (DxDxDG family), efhand (12-mer), both (expanded+efhand). Default: both")
    ap.add_argument("--ef-min-score", type=float, default=3.5, help="Minimum EF-hand 12-mer score (default 3.5)")
    ap.add_argument("--ef-require-gly", action="store_true", help="Require Gly at pos 6±1 for EF-like window")
    ap.add_argument("--ef-ban-proline", action="store_true", help="Disqualify EF-like windows containing Pro")
    args = ap.parse_args()

    analyze(
        in_path=args.in_path,
        outdir=args.outdir,
        groups_path=args.groups,
        user_map_path=args.user_map,
        use_relaxed_alpha=(not args.no_relaxed_alpha),
        write_excel=args.excel,
        tail_buffer=args.tail_buffer,
        tail_fallback_len=args.tail_fallback_len,
        classify_mode=args.classify_mode,
        ca_scan_mode=args.ca_scan_mode,
        ef_min_score=args.ef_min_score,
        ef_require_gly=args.ef_require_gly,
        ef_ban_proline=args.ef_ban_proline,
        motif_library_path=args.motif_library,
        export_motif_library=args.export_default_motif_library,
    )

if __name__ == "__main__":
    main()
