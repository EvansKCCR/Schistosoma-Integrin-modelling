#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schistosome β-integrin regex motif scanner and source-aware classifier
======================================================================

Input:
    FASTA/FA file containing β-integrin candidates or mixed parasite/human controls.

Purpose:
    1. Scan β-integrin candidates against a regex-compatible motif library.
    2. Annotate β-integrin-associated features: βI-like MIDAS 5-mers, NPxY/NPxF
       cytoplasmic-tail motifs, transmembrane/juxtamembrane signatures and
       cysteine-rich I-EGF-like motifs.
    3. Distinguish schistosome β-Int1/ITGB2-like integrins from human β-integrins
       using motif presentation, not sequence ID alone.

Important schistosome ID aliases:
    beta-Int1 = Smp_089700.1
    ITGB2     = MS3_00009865.1_mrna

Outputs:
    beta_integrin_features.csv
    beta_regex_motif_hits.csv
    beta_regex_subgroup_scores.csv
    beta_tail_motifs.csv
    betaI_MIDAS_5mers.csv

Example:
    python beta_integrin_scan_and_classify_regex.py \
        --in beta_integrins.fa \
        --outdir beta_integrin_scan_out \
        --motif-library schisto_beta_integrin_regex_motif_library.tsv \
        --excel
"""

import argparse
import csv
import io
import os
import re
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any

try:
    import pandas as pd
except Exception:
    pd = None

HYDRO = set("AILVFWYMGTC")
AA_RE = re.compile(r"[^A-Z*]")


DEFAULT_LIBRARY_TSV = """motif_id	subgroup	source_class	region	regex	weight	required	diagnostic	max_hits	description
BETA_MIDAS_DXSXS	beta_integrin_general	general	full	D.S.S	1.5	False	accessory	5	Generic βI-like MIDAS 5-mer class DXSXS; annotation support only.
BETA_MIDAS_DXSXT	beta_integrin_general	general	full	D.S.T	1.0	False	accessory	5	Generic βI-like MIDAS 5-mer class DXSXT; annotation support only.
BETA_TAIL_NPxY	beta_integrin_general	general	cterm160	NP.Y	1.5	False	accessory	5	Generic β-integrin PTB-binding NPxY motif in C-terminal/tail region.
BETA_TAIL_NPxF	beta_integrin_general	general	cterm160	NP.F	1.5	False	accessory	5	Generic β-integrin NPxF motif in C-terminal/tail region.
BETA_TM_HDXR_GENERIC	beta_integrin_general	general	cterm220	[LIV]{2,3}.{0,4}[WY][KR].{0,5}(?:HDR|DDR|YDR|SDL)[RK]E	1.0	False	accessory	3	Generic β-integrin transmembrane/juxtamembrane HDxR-like region, expanded to include parasite DDR variants.
BETA_EGF_CYS_GENERIC	beta_integrin_general	general	full	C.{1,4}C.{1,5}C.{1,5}C	0.5	False	accessory	10	Generic cysteine-rich I-EGF-like pattern; broad annotation support only.
P_B01_PSI_YFLTD	Schistosome_beta_Int1_ITGB2_like	parasite	full	YPVDLYFLTDLSYTM	3.0	True	diagnostic	1	Parasite-specific PSI/βI-domain-proximal motif; block 1.
P_B02_GNLDSPEGGMD	Schistosome_beta_Int1_ITGB2_like	parasite	full	GNLDSPEGGMDALLQ	3.0	True	diagnostic	1	Parasite-specific hybrid-domain motif with GMDALLQ presentation; block 2.
P_B03_QMGFGAFVDK	Schistosome_beta_Int1_ITGB2_like	parasite	full	QMGFGAFVDKPVFPF	3.0	True	diagnostic	1	Parasite-specific βI-like GFGAF motif beginning QMG; block 3.
P_B04_CDPPFLYKH	Schistosome_beta_Int1_ITGB2_like	parasite	full	CDPPFLYKHILSLTD	2.5	False	diagnostic	1	Parasite-specific conserved extracellular motif; block 4.
P_B05_CSGRGVCDC	Schistosome_beta_Int1_ITGB2_like	parasite	full	CSGRGVCDCGQCFCN	2.0	False	support	1	Parasite cysteine-rich/I-EGF-like motif; block 5.
P_B06_CSGRGRCVC	Schistosome_beta_Int1_ITGB2_like	parasite	full	CSGRGRCVCGKCKCN	2.0	False	support	1	Parasite cysteine-rich/I-EGF-like motif; block 6.
P_B07_VARCSEIGWR	Schistosome_beta_Int1_ITGB2_like	parasite	full	VARCSE[AI]IGWRAGAR	2.5	False	diagnostic	1	Parasite-specific CSE/IGWRAGAR presentation; block 7.
P_B08_YEGTFCECDR	Schistosome_beta_Int1_ITGB2_like	parasite	full	YEGT[FY]CECDRHGCKR	2.5	False	diagnostic	1	Parasite-specific cysteine-rich motif with DRHGCKR ending; block 8.
P_B09_QVLTEADISV	Schistosome_beta_Int1_ITGB2_like	parasite	full	QVLTEADISVIFAVD	2.5	False	diagnostic	1	Parasite-specific βI/hybrid-domain motif; block 9.
P_B10_LFASDGGFHL	Schistosome_beta_Int1_ITGB2_like	parasite	full	LFASDGGFHLAGDGR	3.0	True	diagnostic	1	Parasite-specific acidic-metal-site-associated motif; block 10.
P_B11_CECKPGYTGDR	Schistosome_beta_Int1_ITGB2_like	parasite	full	CEC[KR]PGYTGDRCDCM	2.0	False	support	1	Parasite cysteine-rich motif; block 11.
P_B12_WHNSDQTDYP	Schistosome_beta_Int1_ITGB2_like	parasite	full	WHNSDQTDYPSVGEI	2.5	False	diagnostic	1	Parasite-specific βI/hybrid-domain motif; block 12.
P_B13_TM_IDDRRE	Schistosome_beta_Int1_ITGB2_like	parasite	cterm220	LLIYKLVITIDDRRE	3.0	True	diagnostic	1	Parasite transmembrane/juxtamembrane motif with IDDRRE instead of canonical HDR/SDL presentation; block 13.
P_B14_IAGLPPPTTC	Schistosome_beta_Int1_ITGB2_like	parasite	full	IAGL[IV][LR]PPPTTCQLT	2.5	False	diagnostic	1	Parasite-specific extracellular motif replacing human PNDG/CHL region; block 14.
P_B15_TAIL_ENPIF	Schistosome_beta_Int1_ITGB2_like	parasite	cterm160	ENMRWEMAENPIFES	3.0	True	diagnostic	1	Parasite β-tail/terminal motif containing ENPIF; block 15.
P_B16_CGECKCQSGY	Schistosome_beta_Int1_ITGB2_like	parasite	full	CGECKCQSGYSGDFC	2.0	False	support	1	Parasite cysteine-rich motif; block 16.
P_B17_MATYDADYLE	Schistosome_beta_Int1_ITGB2_like	parasite	full	MATYDADYLEVQVFS	3.0	False	diagnostic	1	Parasite-only motif in submitted set; block 17.
P_B18_QTACSCPSCE	Schistosome_beta_Int1_ITGB2_like	parasite	full	QTACSCPSCEK[LM]PMP	2.5	False	diagnostic	1	Parasite-only cysteine-rich motif; block 18.
P_B19_NQVCGGPQRG	Schistosome_beta_Int1_ITGB2_like	parasite	full	NQVCGGPQRG[ST]CQCN	2.5	False	diagnostic	1	Parasite-biased cysteine-rich motif; block 19.
P_B20_TAIL_LNPTF	Schistosome_beta_Int1_ITGB2_like	parasite	cterm160	PTTNVLNPTFEENGY	3.0	False	diagnostic	1	Parasite β-tail motif containing LNPTF; block 20.
P_B21_AIYVNRKVDL	Schistosome_beta_Int1_ITGB2_like	parasite	full	AIYVNRKVDLNIIRK	2.5	False	diagnostic	1	Parasite-only motif; block 21.
P_B22_KIYQISIKAK	Schistosome_beta_Int1_ITGB2_like	parasite	full	KI[GN]YQISIKAKRCF	2.5	False	diagnostic	1	Parasite-only motif; block 22.
P_B23_ENEFSKKTVC	Schistosome_beta_Int1_ITGB2_like	parasite	full	ENEFSKKTVCNDHPV	2.5	False	diagnostic	1	Parasite-only motif; block 23.
P_B24_TMIVNFTFQSI	Schistosome_beta_Int1_ITGB2_like	parasite	full	TM[IM]VNFTFQSI	2.0	False	support	1	Parasite N-terminal/extracellular motif; block 24.
P_B25_SLTPFSARCQ	Schistosome_beta_Int1_ITGB2_like	parasite	full	SLTPFSARCQLRGH	2.5	False	diagnostic	1	Parasite-only motif; block 25.
P_B26_KKVNLKMVAL	Schistosome_beta_Int1_ITGB2_like	parasite	full	KKV[NT]LKMVALEDQAV	2.5	False	diagnostic	1	Parasite-only motif; block 26.
P_B27_AAMLAQCTE	Schistosome_beta_Int1_ITGB2_like	parasite	full	A[AT]MLAQCTE[SY]GVKEP	2.5	False	diagnostic	1	Parasite-only motif; block 27.
H_B01_PSI_YILMD	Human_beta_integrin_like	human	full	[SY]P[IV]D[IL]Y[IY]L[MV]D[FLV]S[ANY]SM	2.5	True	diagnostic	1	Human β-integrin PSI/βI-proximal motif; excludes parasite YFLTD presentation.
H_B02_GNLDSPEGGFD	Human_beta_integrin_like	human	full	[AGR]N[ILR]D[AST]PEGG[FL]DA[IM][LM]Q	2.5	True	diagnostic	1	Human β-integrin hybrid-domain motif with FDA/LDA presentation.
H_B03_RIGFGS	Human_beta_integrin_like	human	full	[RT][IL]GFG[AKS][FY]V[DE]K[PTV][SV][LMSV]P[FQY]	2.5	False	diagnostic	1	Human βI-like GFG motif presentation; excludes parasite QMGFGAFVDKPVFPF.
H_B04_CPPFGYKH	Human_beta_integrin_like	human	full	[CS].[PS][MPT][FH][AGS][FY][HIKR][HN][IV][IL][KPST]LT[DEGN]	2.0	False	support	1	Human extracellular motif consensus from block 4.
H_B05_CSGRGDCY	Human_beta_integrin_like	human	full	CS[GNQ][KLNR]G[DEHV]C[LQVY]CG[HKQR]C[ILSV]C[HRSY]	2.0	False	support	1	Human cysteine-rich/I-EGF motif consensus from block 5.
H_B06_CSGRGTCV	Human_beta_integrin_like	human	full	C[NS]G[HR]G.C[EKRV]C[GN][KRSV]C.C[HILT]	2.0	False	support	1	Human cysteine-rich/I-EGF motif consensus from block 6.
H_B07_IGWRNV	Human_beta_integrin_like	human	full	[ATV][AT][ALV]C[ERS].IGWR[KNP][DEV][AST].	2.5	False	diagnostic	1	Human IGWR region with RNV/RND/RKE-like presentation rather than parasite IGWRAGAR.
H_B09_QKLSENNI	Human_beta_integrin_like	human	full	.[AKL]L[AISV].[AHKN]N[IV][ILNQ][LPTV]IFAV[QT]	2.0	False	support	1	Human βI/hybrid-domain motif consensus from block 9.
H_B10_VFATDDGF	Human_beta_integrin_like	human	full	[LV][FV].[ST][DE][ADQS].[FST]H.[AEG][AGLM]D[GS][AKR]	2.5	False	diagnostic	1	Human metal-site-associated motif consensus from block 10; differs from parasite LFASDGGFHLAGDGR.
H_B12_DYPSVGQL	Human_beta_integrin_like	human	full	Y[KSTV].[SY][HNRT][ETVY].[DE][HY]P[ST][ILV][AGP][HLQT][LMV]	2.0	False	support	1	Human DYPS/HPST region consensus from block 12.
H_B13_TM_HDR	Human_beta_integrin_like	human	cterm220	(?:LLIWKLL[IM]I[IH]HDR[KR]E|LCIWKLLVSFHDRKE|LVIWKALIHLSDLRE|VLAYRLSVEIYDRRE)	2.5	True	diagnostic	1	Human transmembrane/juxtamembrane HDR/SDL/YDR presentation; block 13.
H_B14_PNDGCHL	Human_beta_integrin_like	human	full	L[AG][AG]I[FLMV].[PR][NS]D[EG].CH[LV][DEGK]	2.5	False	diagnostic	1	Human PNDG/CHL-like motif; excludes parasite PPPTTCQLT motif.
H_B15_TAIL_NPLY_NPIY	Human_beta_integrin_like	human	cterm160	NPL[YT]|NPIY	2.5	True	diagnostic	3	Human β-tail NPLY/NPIY-like motif presentation from submitted human set.
"""


# -----------------------------------------------------------------------------
# FASTA I/O and ID normalization
# -----------------------------------------------------------------------------

def read_fasta(path: str) -> Dict[str, str]:
    """Read FASTA/FA file. Header ID is the first whitespace-delimited token."""
    seqs: Dict[str, str] = {}
    name = None
    with open(path, "r", encoding="utf-8") as fh:
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
                    raise ValueError("FASTA format error: sequence found before first header")
                seqs[name] += line.strip()
    return {k: clean_sequence(v) for k, v in seqs.items()}


def clean_sequence(seq: str) -> str:
    """Uppercase and remove gaps/spaces/non-AA symbols except stop, then remove stop."""
    seq = seq.upper().replace("-", "").replace(".", "")
    seq = AA_RE.sub("", seq)
    return seq.replace("*", "")


def canonical_id(seq_id: str) -> str:
    """Normalize key schistosome aliases while preserving other IDs."""
    sid = seq_id.strip()
    up = sid.upper()

    # Schistosome beta-Int1 alias
    if up in {"BETA-INT1", "BETA_INT1", "BETAINT1", "SMP_089700.1", "SMP_089700"}:
        return "beta-Int1"

    # Schistosome ITGB2 alias; keep exact ITGB2 from user-provided sheet as parasite alias.
    if up in {"ITGB2", "MS3_00009865.1_MRNA", "MS3_00009865.1", "MS3_00009865"}:
        return "ITGB2"

    return sid


def expected_source_from_id(seq_id: str) -> str:
    """Heuristic source group from sequence ID only. Classification itself is motif-based."""
    sid = seq_id.strip()
    up = sid.upper()

    if sid.startswith("sp|") or "HUMAN" in up or up.endswith("_HUMAN"):
        return "human"
    if canonical_id(sid) in {"beta-Int1", "ITGB2"}:
        return "parasite"
    if up.startswith(("SMP_", "MS3_", "EWB00", "SRAE", "SM.", "SCHISTO")):
        return "parasite"
    return "unknown"


# -----------------------------------------------------------------------------
# Region detection and generic β-integrin feature scans
# -----------------------------------------------------------------------------

def find_hydrophobic_windows(seq: str, window: int = 19, min_hydro: int = 15) -> List[Tuple[int, int]]:
    """Return merged hydrophobic windows as 0-based half-open intervals."""
    hits = []
    for i in range(0, max(0, len(seq) - window + 1)):
        win = seq[i:i + window]
        if sum(1 for aa in win if aa in HYDRO) >= min_hydro:
            hits.append((i, i + window))

    merged: List[List[int]] = []
    for s, e in hits:
        if not merged or s > merged[-1][1] + 2:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    return [(s, e) for s, e in merged]


def tail_region(seq: str, fallback_len: int = 160) -> Tuple[int, str, str]:
    """
    Estimate β-integrin C-terminal cytoplasmic/tail-containing region.

    Unlike the older script, this does not require human-like HDR. It handles
    schistosome IDDR/DDR-like juxtamembrane presentations as well as SDL/YDR.
    Returns (start_1based, region_sequence, method).
    """
    L = len(seq)
    cterm = seq[max(0, L - 260):]

    # Prefer the last predicted transmembrane segment if it occurs near the C-terminus.
    tms = find_hydrophobic_windows(seq)
    near_cterm_tms = [(s, e) for s, e in tms if e > L - 260]
    if near_cterm_tms:
        s, e = near_cterm_tms[-1]
        # include a few residues before the membrane end so IDDR/HDR-adjacent motifs are retained
        start = max(0, e - 8)
        return start + 1, seq[start:], "last_hydrophobic_TM_window"

    # Fallback: detect juxtamembrane HDR/DDR/YDR/SDL variants in the last 260 aa.
    # Keep 40 aa upstream to retain the end of the transmembrane segment.
    jm = list(re.finditer(r"(?:HDR|DDR|YDR|SDL)[RK]E", cterm))
    if jm:
        m = jm[-1]
        start = max(0, (L - len(cterm)) + m.start() - 40)
        return start + 1, seq[start:], "juxtamembrane_HDR_DDR_YDR_SDL"

    start = max(0, L - fallback_len)
    return start + 1, seq[start:], f"last_{fallback_len}_aa_fallback"


def region_sequence(seq: str, region: str, tail_cache: Optional[Tuple[int, str, str]] = None) -> Tuple[int, str]:
    """Return (offset_1based, region_sequence) for motif scanning."""
    L = len(seq)
    region = region.lower().strip()

    if region in {"full", "sequence", "all"}:
        return 1, seq

    if region.startswith("cterm"):
        n = int(re.sub(r"[^0-9]", "", region) or "160")
        start = max(0, L - n)
        return start + 1, seq[start:]

    if region == "tail":
        if tail_cache is None:
            tail_cache = tail_region(seq)
        return tail_cache[0], tail_cache[1]

    # Unknown region: scan full sequence but signal in output via region column.
    return 1, seq


def scan_beta_tail(seq: str) -> List[Dict[str, Any]]:
    """Scan the inferred tail-containing region for NPxY and NPxF motifs."""
    tstart, tail, method = tail_region(seq)
    rows: List[Dict[str, Any]] = []

    for cls, pattern in [("NPxY", r"NP.Y"), ("NPxF", r"NP.F")]:
        for m in re.finditer(pattern, tail):
            pos0 = m.start()
            flank_left = max(0, pos0 - 6)
            flank_right = min(len(tail), pos0 + 4 + 6)
            flank = tail[flank_left:pos0] + tail[pos0 + 4:flank_right]
            st_count = sum(1 for aa in flank if aa in "ST")
            rows.append({
                "motif_class": cls,
                "motif": m.group(),
                "abs_pos_1based": tstart + pos0,
                "st_flank_STcount_pm6": st_count,
                "tail_start_1based": tstart,
                "tail_method": method,
            })
    return rows


def scan_midas_5mers(seq: str) -> List[Dict[str, Any]]:
    """Scan full sequence for βI-like DXSXS/DXSXT/DXSX* 5-mers."""
    rows: List[Dict[str, Any]] = []
    for i in range(0, max(0, len(seq) - 4)):
        five = seq[i:i + 5]
        if len(five) == 5 and five[0] == "D" and five[2] == "S":
            if five[4] == "S":
                cls = "DXSXS"
            elif five[4] == "T":
                cls = "DXSXT"
            else:
                cls = "DXSX*"
            rows.append({"midas_class": cls, "motif": five, "abs_pos_1based": i + 1})
    return rows


# -----------------------------------------------------------------------------
# Motif library loading and regex scanning
# -----------------------------------------------------------------------------

def _bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y", "required"}


def load_motif_library(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load motif library from TSV, or built-in library if path is omitted."""
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    else:
        text = DEFAULT_LIBRARY_TSV

    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    required_cols = {"motif_id", "subgroup", "source_class", "region", "regex", "weight",
                     "required", "diagnostic", "max_hits", "description"}
    if not reader.fieldnames:
        raise ValueError("Motif library appears empty or lacks a header line.")
    missing = required_cols - set(reader.fieldnames)
    if missing:
        raise ValueError(f"Motif library missing required columns: {sorted(missing)}")

    rows: List[Dict[str, Any]] = []
    seen = set()
    for r in reader:
        if not r.get("motif_id"):
            continue
        motif_id = r["motif_id"].strip()
        if motif_id in seen:
            raise ValueError(f"Duplicate motif_id in library: {motif_id}")
        seen.add(motif_id)
        try:
            re.compile(r["regex"])
        except re.error as e:
            raise ValueError(f"Invalid regex for motif_id {motif_id}: {e}") from e

        row = {
            "motif_id": motif_id,
            "subgroup": r["subgroup"].strip(),
            "source_class": r["source_class"].strip().lower(),
            "region": r["region"].strip(),
            "regex": r["regex"].strip(),
            "weight": float(r["weight"]),
            "required": _bool(r["required"]),
            "diagnostic": r["diagnostic"].strip().lower(),
            "max_hits": int(float(r["max_hits"] or 1)),
            "description": r["description"].strip(),
        }
        rows.append(row)

    return rows


def scan_regex_library(seq_id: str, seq: str, library: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Scan one sequence against the motif library and return motif-hit rows."""
    tail_cache = tail_region(seq)
    hits: List[Dict[str, Any]] = []

    for motif in library:
        offset, subseq = region_sequence(seq, motif["region"], tail_cache)
        pattern = re.compile(motif["regex"])
        matches = list(pattern.finditer(subseq))
        if not matches:
            continue

        for m in matches[:motif["max_hits"]]:
            hits.append({
                "seq_id": seq_id,
                "canonical_id": canonical_id(seq_id),
                "motif_id": motif["motif_id"],
                "subgroup": motif["subgroup"],
                "source_class": motif["source_class"],
                "region": motif["region"],
                "regex": motif["regex"],
                "motif_match": m.group(),
                "abs_pos_1based": offset + m.start(),
                "weight": motif["weight"],
                "required": motif["required"],
                "diagnostic": motif["diagnostic"],
                "description": motif["description"],
            })

    return hits


def summarize_scores(hits: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, int], Dict[str, int]]:
    """
    Collapse motif hits by unique motif_id before scoring, preventing repeated
    sequence copies of one motif from dominating classification.
    """
    best_by_motif: Dict[str, Dict[str, Any]] = {}
    for h in hits:
        mid = h["motif_id"]
        if mid not in best_by_motif:
            best_by_motif[mid] = h

    score_by_subgroup = defaultdict(float)
    req_by_subgroup = defaultdict(int)
    diag_by_subgroup = defaultdict(int)
    source_by_subgroup = {}

    for h in best_by_motif.values():
        sg = h["subgroup"]
        score_by_subgroup[sg] += float(h["weight"])
        source_by_subgroup[sg] = h["source_class"]
        if h["required"]:
            req_by_subgroup[sg] += 1
        if str(h["diagnostic"]).lower() == "diagnostic":
            diag_by_subgroup[sg] += 1

    rows = []
    for sg in sorted(score_by_subgroup.keys()):
        rows.append({
            "subgroup": sg,
            "source_class": source_by_subgroup.get(sg, ""),
            "score": round(score_by_subgroup[sg], 3),
            "required_hits": req_by_subgroup.get(sg, 0),
            "diagnostic_hits": diag_by_subgroup.get(sg, 0),
        })

    return rows, dict(score_by_subgroup), dict(req_by_subgroup), dict(diag_by_subgroup)


# -----------------------------------------------------------------------------
# Classification logic
# -----------------------------------------------------------------------------

def classify_sequence(
    seq_id: str,
    seq: str,
    score_by_subgroup: Dict[str, float],
    req_by_subgroup: Dict[str, int],
    diag_by_subgroup: Dict[str, int],
    tail_rows: List[Dict[str, Any]],
    midas_rows: List[Dict[str, Any]],
    min_parasite_score: float = 10.0,
    min_human_score: float = 8.0,
    score_delta: float = 3.0,
) -> Tuple[str, str]:
    """Return final_class and concise reason."""
    parasite_group = "Schistosome_beta_Int1_ITGB2_like"
    human_group = "Human_beta_integrin_like"
    general_group = "beta_integrin_general"

    p_score = float(score_by_subgroup.get(parasite_group, 0.0))
    h_score = float(score_by_subgroup.get(human_group, 0.0))
    g_score = float(score_by_subgroup.get(general_group, 0.0))

    p_req = int(req_by_subgroup.get(parasite_group, 0))
    h_req = int(req_by_subgroup.get(human_group, 0))
    p_diag = int(diag_by_subgroup.get(parasite_group, 0))
    h_diag = int(diag_by_subgroup.get(human_group, 0))

    npx = len(tail_rows)
    midas = len(midas_rows)
    has_beta_arch = (g_score >= 2.0) or (npx > 0) or (midas > 0)

    # Strong calls require both adequate score and diagnostic/required motif support.
    if p_score >= min_parasite_score and p_req >= 2 and p_score >= h_score + score_delta:
        return (
            "Schistosome β-integrin-like (β-Int1/ITGB2 clade)",
            f"parasite_score={p_score:.1f}, human_score={h_score:.1f}, parasite_required={p_req}, parasite_diagnostic={p_diag}"
        )

    if p_score >= (min_parasite_score - 3.0) and p_req >= 1 and p_score >= h_score + 2.0:
        return (
            "Probable schistosome β-integrin-like (β-Int1/ITGB2 clade)",
            f"parasite-biased motif presentation; parasite_score={p_score:.1f}, human_score={h_score:.1f}, parasite_required={p_req}"
        )

    if h_score >= min_human_score and h_req >= 2 and h_score >= p_score + score_delta:
        return (
            "Human β-integrin-like",
            f"human_score={h_score:.1f}, parasite_score={p_score:.1f}, human_required={h_req}, human_diagnostic={h_diag}"
        )

    if h_score >= (min_human_score - 2.0) and h_req >= 1 and h_score >= p_score + 2.0:
        return (
            "Probable human β-integrin-like",
            f"human-biased motif presentation; human_score={h_score:.1f}, parasite_score={p_score:.1f}, human_required={h_req}"
        )

    if p_score >= 7.0 and h_score >= 6.0 and abs(p_score - h_score) < 2.0:
        return (
            "Ambiguous/mixed β-integrin-like",
            f"similar parasite and human motif scores; parasite_score={p_score:.1f}, human_score={h_score:.1f}"
        )

    if has_beta_arch:
        return (
            "β-integrin-like, unresolved source/class",
            f"generic β-integrin evidence present; MIDAS_5mers={midas}, tail_NPx={npx}, generic_score={g_score:.1f}"
        )

    return (
        "Unassigned",
        f"insufficient β-integrin motif evidence; parasite_score={p_score:.1f}, human_score={h_score:.1f}, generic_score={g_score:.1f}"
    )


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def analyze(
    fasta_path: str,
    outdir: str,
    motif_library_path: Optional[str] = None,
    excel: bool = False,
    min_parasite_score: float = 10.0,
    min_human_score: float = 8.0,
    score_delta: float = 3.0,
    plots: bool = False,
) -> None:
    os.makedirs(outdir, exist_ok=True)

    seqs = read_fasta(fasta_path)
    library = load_motif_library(motif_library_path)

    all_hits: List[Dict[str, Any]] = []
    all_score_rows: List[Dict[str, Any]] = []
    all_tail_rows: List[Dict[str, Any]] = []
    all_midas_rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []

    for sid, seq in seqs.items():
        cid = canonical_id(sid)
        expected = expected_source_from_id(sid)
        tail_start, tail_seq, tail_method = tail_region(seq)
        tm_windows = find_hydrophobic_windows(seq)

        regex_hits = scan_regex_library(sid, seq, library)
        score_rows, scores, reqs, diags = summarize_scores(regex_hits)

        tail_rows = scan_beta_tail(seq)
        midas_rows = scan_midas_5mers(seq)

        final_class, reason = classify_sequence(
            sid, seq, scores, reqs, diags, tail_rows, midas_rows,
            min_parasite_score=min_parasite_score,
            min_human_score=min_human_score,
            score_delta=score_delta,
        )

        parasite_group = "Schistosome_beta_Int1_ITGB2_like"
        human_group = "Human_beta_integrin_like"
        general_group = "beta_integrin_general"

        npxy = sum(1 for r in tail_rows if r["motif_class"] == "NPxY")
        npxf = sum(1 for r in tail_rows if r["motif_class"] == "NPxF")
        st_vals = [r["st_flank_STcount_pm6"] for r in tail_rows if r["motif_class"] == "NPxF"]
        dxsxs = sum(1 for r in midas_rows if r["midas_class"] == "DXSXS")
        dxsxt = sum(1 for r in midas_rows if r["midas_class"] == "DXSXT")
        dxsx_other = sum(1 for r in midas_rows if r["midas_class"] == "DXSX*")

        top_motifs = sorted(
            {h["motif_id"]: h for h in regex_hits}.values(),
            key=lambda x: (-float(x["weight"]), x["motif_id"])
        )[:8]

        mismatch = ""
        if expected in {"parasite", "human"}:
            if expected == "parasite" and "Human β-integrin" in final_class:
                mismatch = "ID_motif_discordance"
            elif expected == "human" and "Schistosome" in final_class:
                mismatch = "ID_motif_discordance"

        feature_rows.append({
            "seq_id": sid,
            "canonical_id": cid,
            "expected_source_from_id": expected,
            "length": len(seq),
            "tail_start_1based": tail_start,
            "tail_method": tail_method,
            "TM_window_count": len(tm_windows),
            "NPxY_count_tail": npxy,
            "NPxF_count_tail": npxf,
            "NPxF_ST_flank_mean_pm6": round(sum(st_vals) / len(st_vals), 3) if st_vals else 0.0,
            "MIDAS_DXSXS_count": dxsxs,
            "MIDAS_DXSXT_count": dxsxt,
            "MIDAS_DXSXstar_count": dxsx_other,
            "regex_motif_hit_count": len(regex_hits),
            "parasite_score": round(float(scores.get(parasite_group, 0.0)), 3),
            "human_score": round(float(scores.get(human_group, 0.0)), 3),
            "general_beta_score": round(float(scores.get(general_group, 0.0)), 3),
            "parasite_required_hits": reqs.get(parasite_group, 0),
            "human_required_hits": reqs.get(human_group, 0),
            "parasite_diagnostic_hits": diags.get(parasite_group, 0),
            "human_diagnostic_hits": diags.get(human_group, 0),
            "final_class": final_class,
            "classification_reason": reason,
            "id_motif_discordance_flag": mismatch,
            "top_weighted_motifs": ";".join([h["motif_id"] for h in top_motifs]),
        })

        for h in regex_hits:
            all_hits.append(h)

        for sr in score_rows:
            sr2 = {"seq_id": sid, "canonical_id": cid, **sr}
            all_score_rows.append(sr2)

        for r in tail_rows:
            all_tail_rows.append({"seq_id": sid, "canonical_id": cid, **r})

        for r in midas_rows:
            all_midas_rows.append({"seq_id": sid, "canonical_id": cid, **r})

    # CSV writing
    def write_csv(path: str, rows: List[Dict[str, Any]], preferred_cols: Optional[List[str]] = None) -> None:
        if not rows:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                fh.write("")
            return
        cols = preferred_cols or list(rows[0].keys())
        # include any extra keys
        for r in rows:
            for k in r.keys():
                if k not in cols:
                    cols.append(k)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    feature_cols = [
        "seq_id", "canonical_id", "expected_source_from_id", "length",
        "tail_start_1based", "tail_method", "TM_window_count",
        "NPxY_count_tail", "NPxF_count_tail", "NPxF_ST_flank_mean_pm6",
        "MIDAS_DXSXS_count", "MIDAS_DXSXT_count", "MIDAS_DXSXstar_count",
        "regex_motif_hit_count", "parasite_score", "human_score", "general_beta_score",
        "parasite_required_hits", "human_required_hits", "parasite_diagnostic_hits", "human_diagnostic_hits",
        "final_class", "classification_reason", "id_motif_discordance_flag", "top_weighted_motifs"
    ]

    write_csv(os.path.join(outdir, "beta_integrin_features.csv"), feature_rows, feature_cols)
    write_csv(os.path.join(outdir, "beta_regex_motif_hits.csv"), all_hits)
    write_csv(os.path.join(outdir, "beta_regex_subgroup_scores.csv"), all_score_rows)
    write_csv(os.path.join(outdir, "beta_tail_motifs.csv"), all_tail_rows)
    write_csv(os.path.join(outdir, "betaI_MIDAS_5mers.csv"), all_midas_rows)

    # Also save the active motif library used.
    with open(os.path.join(outdir, "motif_library_used.tsv"), "w", encoding="utf-8") as fh:
        if motif_library_path:
            with open(motif_library_path, "r", encoding="utf-8") as src:
                fh.write(src.read())
        else:
            fh.write(DEFAULT_LIBRARY_TSV)

    # Optional Excel workbook
    if excel:
        if pd is None:
            print("[WARN] pandas is unavailable; skipped Excel output.", file=sys.stderr)
        else:
            xlsx_path = os.path.join(outdir, "beta_integrin_regex_classification.xlsx")
            try:
                with pd.ExcelWriter(xlsx_path) as writer:
                    pd.DataFrame(feature_rows).to_excel(writer, sheet_name="features", index=False)
                    pd.DataFrame(all_hits).to_excel(writer, sheet_name="regex_hits", index=False)
                    pd.DataFrame(all_score_rows).to_excel(writer, sheet_name="scores", index=False)
                    pd.DataFrame(all_tail_rows).to_excel(writer, sheet_name="tail_motifs", index=False)
                    pd.DataFrame(all_midas_rows).to_excel(writer, sheet_name="MIDAS_5mers", index=False)
                    pd.DataFrame(load_motif_library(motif_library_path)).to_excel(writer, sheet_name="motif_library", index=False)
            except Exception as e:
                print(f"[WARN] Could not write Excel workbook: {e}", file=sys.stderr)

    if plots:
        write_plots(outdir, feature_rows)

    print(f"[OK] Wrote beta-integrin classification outputs to: {outdir}")
    print(f" - {os.path.join(outdir, 'beta_integrin_features.csv')}")
    print(f" - {os.path.join(outdir, 'beta_regex_motif_hits.csv')}")
    print(f" - {os.path.join(outdir, 'beta_regex_subgroup_scores.csv')}")


def write_plots(outdir: str, feature_rows: List[Dict[str, Any]]) -> None:
    """Optional quick diagnostic plots. Skips gracefully if matplotlib is missing."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        print("[WARN] matplotlib unavailable; skipped plots.", file=sys.stderr)
        return

    if not feature_rows:
        return

    labels = [r["canonical_id"] for r in feature_rows]
    p = [float(r["parasite_score"]) for r in feature_rows]
    h = [float(r["human_score"]) for r in feature_rows]
    g = [float(r["general_beta_score"]) for r in feature_rows]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.65), 4), dpi=180)
    ax.bar(x - width, p, width, label="parasite")
    ax.bar(x, h, width, label="human")
    ax.bar(x + width, g, width, label="general β")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Collapsed regex motif score")
    ax.set_title("β-integrin regex classification scores")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "beta_regex_classification_scores.png"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Regex motif scanning and source-aware classification of schistosome/human β-integrins."
    )
    ap.add_argument("--in", dest="fasta_path", required=True, help="Input FASTA/FA file.")
    ap.add_argument("--outdir", required=True, help="Output directory.")
    ap.add_argument("--motif-library", help="Optional TSV motif library. If omitted, built-in library is used.")
    ap.add_argument("--excel", action="store_true", help="Also write beta_integrin_regex_classification.xlsx.")
    ap.add_argument("--plots", action="store_true", help="Write quick score summary PNG.")
    ap.add_argument("--min-parasite-score", type=float, default=10.0, help="High-confidence parasite score threshold.")
    ap.add_argument("--min-human-score", type=float, default=8.0, help="High-confidence human score threshold.")
    ap.add_argument("--score-delta", type=float, default=3.0, help="Minimum score separation for high-confidence source calls.")
    args = ap.parse_args()

    analyze(
        fasta_path=args.fasta_path,
        outdir=args.outdir,
        motif_library_path=args.motif_library,
        excel=args.excel,
        min_parasite_score=args.min_parasite_score,
        min_human_score=args.min_human_score,
        score_delta=args.score_delta,
        plots=args.plots,
    )


if __name__ == "__main__":
    main()
