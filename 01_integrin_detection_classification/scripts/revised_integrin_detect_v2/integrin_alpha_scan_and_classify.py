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

def classify_from_features(metrics: dict) -> (str, str):
    """Return (label, reason) derived strictly from features present.
    Precedence:
      1) Parasite HENLA–DIDGDGID: HENLA>0 & DIDGDGID>0
      2) RGD-binding–like: RGD present & any α-tail hallmark/variant present
      3) Laminin-binding–like, mucin-rich: no RGD & NXS/T density ≥ 1.0 per 100 aa
      4) Else: Unassigned non-I (from features)
    """
    L = max(1, int(metrics.get('Length', 0)))
    nxst = float(metrics.get('NXS/T_count', 0))
    nxst_per100 = 100.0 * nxst / L
    tail_hits = sum(int(metrics.get(k, 0)) for k in (
        'GFFKR_count','GFFRS_count','GFFRR_count','GFF_relaxed_count','GFFKR_family_expanded_count'
    ))
    has_tail = tail_hits > 0
    rgd   = bool(metrics.get('RGD_flag', False))
    henla = int(metrics.get('HENLA_count', 0)) > 0
    didg  = int(metrics.get('DIDGDGID_count', 0)) > 0

    if henla and didg:
        return 'Non-I (parasite HENLA–DIDGDGID subclade)', 'HENLA>0 & DIDGDGID>0'
    if rgd and has_tail:
        return 'Non-I (RGD-binding–like)', 'RGD present & α-tail hallmark/variant present'
    if (not rgd) and nxst_per100 >= 1.0:
        return 'Non-I (laminin-binding–like, mucin-rich)', f'NXS/T density {nxst_per100:.2f} ≥ 1.0 per 100 aa & no RGD'
    return 'Unassigned non-I (from features)', 'No rule matched; consider phylogeny/HMM'

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
            ca_scan_mode: str = 'both', ef_min_score: float = 3.5, ef_require_gly: bool = False, ef_ban_proline: bool = False):
    os.makedirs(outdir, exist_ok=True)
    aligned = read_msa_auto(in_path)
    raw = ungap_upper(aligned)

    gmap = load_groups(groups_path)
    groups = sorted(set(gmap.values())) if gmap else ["all"]

    submap = load_subgroup_mapping(user_map_path)

    tail_rows: List[List[object]] = []
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
    ap.add_argument("--classify-mode", choices=["features","map","none"], default="features", help="How to assign subgroup labels: features (rules), map (from --map), or none (blank)")
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
    )

if __name__ == "__main__":
    main()
