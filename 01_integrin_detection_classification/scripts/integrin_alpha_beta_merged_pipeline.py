#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""integrin_alpha_beta_merged_pipeline_patching.py

Merged α/β integrin pipeline with:
  1) α/β auto-routing hardened against α→β misclassification (TM-anchored tail; β-tail length constraint;
     TM-proximal α GFF hallmark override).
  2) Existing α pipeline execution: integrin_alpha_scan_and_classify.py
  3) Existing β pipeline execution: cobalt_integrin_beta_panels.py
  4) β functional-group classification (tail PTB + MIDAS composition) (as before).
  5) NEW: Pairing-likelihood classification (separate from functional-group): uses TM, SP, TM-proximal GFF motif,
     and FG-GAP repeat count (>=2) to call whether an α is likely competent for β-integrin pairing.

Outputs (under --outdir)
------------------------
alpha/   : outputs from integrin_alpha_scan_and_classify.py
beta/    : outputs from cobalt_integrin_beta_panels.py
merged/  :
   - merged_integrin_classification.csv
   - merged_summary_counts.png
   - routing_debug.csv
   - pairing_ability.csv   (NEW)

Notes
-----
• This script is designed to be run from a folder containing the two pipeline scripts, OR you can pass explicit
  paths with --alpha-script and --beta-script.
• Pairing-likelihood is a *separate* heuristic from functional-group classification and is intended to support
  biological interpretation of α–β pairing competence.
"""

import argparse
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import importlib.util

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# FASTA / NEXUS-MATRIX parsers
# -----------------------------

def read_fasta(path: str) -> Dict[str, str]:
    seqs: Dict[str, str] = {}
    name = None
    with open(path, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                name = line[1:].strip().split()[0]
                if name in seqs:
                    raise ValueError(f'Duplicate FASTA ID: {name}')
                seqs[name] = ''
            else:
                if name is None:
                    raise ValueError('FASTA format error: sequence line before first header')
                seqs[name] += line
    return seqs


def read_nexus_matrix(path: str) -> Dict[str, str]:
    seqs: Dict[str, str] = defaultdict(str)
    inside = False
    with open(path, 'r') as fh:
        for line in fh:
            raw = line.rstrip('\n')
            up = raw.strip().upper()
            if not inside:
                if up.startswith('MATRIX'):
                    inside = True
                continue
            if raw.strip().startswith(';'):
                break
            if not raw.strip():
                continue
            parts = raw.split()
            if len(parts) < 2:
                continue
            sid = parts[0]
            chunk = ''.join(parts[1:])
            seqs[sid] += chunk
    return dict(seqs)


def read_msa_auto(path: str) -> Dict[str, str]:
    with open(path, 'r') as fh:
        head = ''.join([next(fh, '') for _ in range(5)])
    if '#NEXUS' in head.upper():
        return read_nexus_matrix(path)
    return read_fasta(path)


def ungap_upper(seqs: Dict[str, str]) -> Dict[str, str]:
    return {k: v.replace('-', '').upper() for k, v in seqs.items()}


def write_fasta(seqs: Dict[str, str], path: str, wrap: int = 80):
    with open(path, 'w') as fh:
        for sid, seq in seqs.items():
            fh.write(f'>{sid}\n')
            for i in range(0, len(seq), wrap):
                fh.write(seq[i:i+wrap] + '\n')


# -----------------------------
# Group mapping
# -----------------------------

def load_groups(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    m: Dict[str, str] = {}
    with open(path, 'r') as fh:
        for line in fh:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue
            sid, grp = parts[0], parts[1]
            m[sid] = grp
    return m


def heuristic_group(sid: str) -> str:
    if 'HUMAN' in sid.upper() or sid.startswith('sp|'):
        return 'human'
    return 'parasite'


# -----------------------------
# α/β routing (hardened)
# -----------------------------

MAX_BETA_TAIL_LEN = 120
TM_WINDOW = 19
TM_HYDRO_THRESHOLD = 17
HDR_MAX_OFFSET_FROM_TM = 80

# Expanded α-tail hallmark (covers parasite variants like GFFHRK)
_ALPHA_TAIL_RE = re.compile(r'GFF[KRHATDNQS][KRH]')

# β-tail PTB motifs
_BETA_NPXY_RE = re.compile(r'NP.Y')
_BETA_NPXF_RE = re.compile(r'NP.F')

# MIDAS-like & ADMIDAS-like heuristics
_MIDAS_5MER_RE = re.compile(r'D.S.S|D.S.T')
_ADMIDAS_RE = re.compile(r'E..D...D')

HYDRO_TAIL = set('AILVFWYMGTC')

# FG–GAP proximity proxy
FG_WINDOW = 30

def count_fggap(seq_u: str) -> int:
    cnt = 0
    fg_pos = [m.start() for m in re.finditer('FG', seq_u)]
    gap_pos = [m.start() for m in re.finditer('GAP', seq_u)]
    j = 0
    for i in fg_pos:
        while j < len(gap_pos) and gap_pos[j] < i:
            j += 1
        k = j
        while k < len(gap_pos) and gap_pos[k] - i <= FG_WINDOW:
            cnt += 1
            k += 1
    return cnt


def last_tm_end(seq_u: str) -> Tuple[Optional[int], bool]:
    """Return (tm_end_0based_exclusive, found) for the last TM-like 19-mer."""
    best = None
    for i in range(len(seq_u) - (TM_WINDOW - 1)):
        window = seq_u[i:i+TM_WINDOW]
        if sum(1 for c in window if c in HYDRO_TAIL) >= TM_HYDRO_THRESHOLD:
            best = i + TM_WINDOW
    return best, (best is not None)


def beta_tail_region(seq_u: str) -> Tuple[int, str, bool]:
    """Return (tail_start_0based, tail_seq, tm_found). Tail anchored to last TM-like segment."""
    tm_end, tm_found = last_tm_end(seq_u)
    if tm_found:
        tail_start = tm_end
        m = re.search('HDR', seq_u[tm_end:tm_end + HDR_MAX_OFFSET_FROM_TM])
        if m:
            tail_start = tm_end + m.end()
        return tail_start, seq_u[tail_start:], True
    start = max(0, len(seq_u) - 200)
    return start, seq_u[start:], False


def routing_evidence(seq_u: str) -> dict:
    tail_win = seq_u[-200:]
    alpha_tail_hits = len(_ALPHA_TAIL_RE.findall(tail_win))
    fggap = count_fggap(seq_u)

    tail_start, tail_seq, tm_found = beta_tail_region(seq_u)
    beta_tail_len = len(tail_seq)

    # α hallmark can appear immediately after TM in the cytosolic tail
    alpha_gff_in_tail = 1 if re.search(r'GFF[KRHATDNQS][KRH]', tail_seq[:25]) else 0

    beta_ptb_npxy = len(_BETA_NPXY_RE.findall(tail_seq))
    beta_ptb_npxf = len(_BETA_NPXF_RE.findall(tail_seq))
    beta_ptb = beta_ptb_npxy + beta_ptb_npxf

    midas = len(_MIDAS_5MER_RE.findall(seq_u))
    admidas = len(_ADMIDAS_RE.findall(seq_u))

    return {
        'alpha_tail_hits': alpha_tail_hits,
        'alpha_gff_in_tail': alpha_gff_in_tail,
        'fggap_count': fggap,
        'beta_ptb_total': beta_ptb,
        'beta_ptb_npxy': beta_ptb_npxy,
        'beta_ptb_npxf': beta_ptb_npxf,
        'beta_tail_start0': tail_start,
        'beta_tail_len': beta_tail_len,
        'tm_found': int(tm_found),
        'midas_like_hits': midas,
        'admidas_like_hits': admidas,
    }


def detect_subunit(seq_u: str) -> Tuple[str, dict]:
    """Return (call, evidence) with α-protection against PTB+MIDAS mimicry."""
    ev = routing_evidence(seq_u)

    alpha_tail_hits = ev['alpha_tail_hits']
    alpha_gff_in_tail = int(ev.get('alpha_gff_in_tail', 0))
    fggap = ev['fggap_count']

    beta_ptb = ev['beta_ptb_total']
    beta_tail_len = ev['beta_tail_len']

    midas = ev['midas_like_hits']
    admidas = ev['admidas_like_hits']

    beta_tail_valid = (beta_ptb > 0) and (beta_tail_len <= MAX_BETA_TAIL_LEN)

    # α GFF override: fixes α sequences with occasional single PTB + MIDAS-like patterns
    if alpha_gff_in_tail > 0 and beta_ptb <= 1:
        ev['beta_tail_valid'] = int(beta_tail_valid)
        ev['alpha_gff_override'] = 1
        return 'alpha', ev
    ev['alpha_gff_override'] = 0

    alpha_strong = (alpha_tail_hits > 0) or (alpha_gff_in_tail > 0) or (fggap >= 2)
    alpha_moderate = (fggap >= 1) or (alpha_tail_hits > 0) or (alpha_gff_in_tail > 0)

    beta_strong = beta_tail_valid and (midas > 0 or admidas > 0)

    alpha_score = (4 * alpha_tail_hits) + (4 if alpha_gff_in_tail > 0 else 0) + (3 if fggap >= 2 else 0) + min(4, fggap // 2)
    beta_score = (5 if beta_tail_valid else 0) + (2 if (beta_tail_valid and midas > 0) else 0) + (1 if (beta_tail_valid and admidas > 0) else 0)

    ev['alpha_score'] = alpha_score
    ev['beta_score'] = beta_score
    ev['beta_tail_valid'] = int(beta_tail_valid)
    ev['alpha_strong'] = int(alpha_strong)
    ev['beta_strong'] = int(beta_strong)

    if alpha_strong and not beta_strong:
        return 'alpha', ev
    if beta_strong and not alpha_strong:
        if beta_ptb == 1 and alpha_moderate:
            return 'alpha', ev
        return 'beta', ev

    if alpha_strong and beta_strong:
        if alpha_tail_hits > 0 or alpha_gff_in_tail > 0:
            return 'alpha', ev
        return ('alpha' if alpha_score >= beta_score else 'beta'), ev

    if beta_tail_valid and (alpha_tail_hits == 0) and (alpha_gff_in_tail == 0) and (fggap < 2):
        if beta_ptb == 1 and alpha_moderate:
            return 'alpha', ev
        return 'beta', ev

    if alpha_tail_hits > 0 or alpha_gff_in_tail > 0 or fggap >= 1:
        return 'alpha', ev

    return 'unknown', ev


# -----------------------------
# Pairing-likelihood (UPDATED; separate from functional-group classification)
# -----------------------------
#
# Replaces the brittle binary rule (TM+SP+GFF+FG-GAP>=2) with a weighted score.
# Keeps all existing evidence fields but makes calls robust to motif drift and
# taxa where SP prediction is weaker (e.g., some parasites).

HYDRO_SP = set('AILMVFWY')

def has_signal_peptide(seq_u: str, n_first: int = 30, min_run: int = 7) -> bool:
    """Simple SP heuristic: hydrophobic run in first n_first aa."""
    n = seq_u[:n_first]
    run = maxrun = 0
    for c in n:
        if c in HYDRO_SP:
            run += 1
            if run > maxrun:
                maxrun = run
        else:
            run = 0
    return maxrun >= min_run


def find_tm_proximal_gff(tail_seq: str, max_scan: int = 30) -> str:
    """Return e.g. 'GFFHR'/'GFFKR' if present near TM in tail, else ''.

    Variant-tolerant: GFF + (K/R/H) + (A/T/D/N/Q/S) + (K/R/H)
    """
    tail_seq = (tail_seq or '').upper()
    m = re.search(r'(GFF[KRH][ATDNQS][KRH])', tail_seq[:max_scan])
    if not m:
        return ''
    return m.group(1)[:5]


def pairing_score(tm_found: bool, sp_found: bool, gff_found: bool, fggap_count: int) -> int:
    """Weighted evidence score for alpha-beta pairing competence.

    Evidence weights (simple & explainable):
      TM present:        +2
      Signal peptide:    +2
      TM-proximal GFF:   +2
      Each FG–GAP proxy: +1 (uses your existing count_fggap)
    """
    score = 0
    if tm_found:
        score += 2
    if sp_found:
        score += 2
    if gff_found:
        score += 2
    score += int(fggap_count)
    return score


def pairing_label_from_score(score: int) -> str:
    """Convert score to a 3-level call."""
    if score >= 6:
        return '🟢 HIGH'
    if score >= 4:
        return '🟡 MEDIUM'
    return '🔴 LOW'


def pairing_ability_call(tm_found: bool, sp_found: bool, gff_found: bool, fggap_count: int) -> str:
    """Backward-compatible wrapper returning a label.

    Signature preserved so existing pipeline code keeps working.
    """
    score = pairing_score(tm_found, sp_found, gff_found, fggap_count)
    return pairing_label_from_score(score)

# -----------------------------
# Dynamic import
# -----------------------------

def import_from_path(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot import {module_name} from {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


# -----------------------------
# β functional-group classification
# -----------------------------

@dataclass
class BetaClass:
    label: str
    reason: str


def classify_beta_row(row: pd.Series, st_mean: float) -> BetaClass:
    npxy = int(row.get('NPxY_count_tail', 0))
    npxf = int(row.get('NPxF_count_tail', 0))
    dxsxs = int(row.get('DXSXS_count', 0))
    dxsxt = int(row.get('DXSXT_count', 0))

    if npxy > 0 and npxf == 0:
        tail_lab = 'β-tail: NPxY-dominant'
    elif npxf > 0 and npxy == 0:
        tail_lab = 'β-tail: NPxF-dominant'
    elif npxf > 0 and npxy > 0:
        tail_lab = 'β-tail: mixed NPxY/NPxF'
    else:
        tail_lab = 'β-tail: no PTB motif detected'

    if dxsxs > 0 and dxsxt == 0:
        mid_lab = 'βI MIDAS: DXSXS-only'
    elif dxsxt > 0 and dxsxs == 0:
        mid_lab = 'βI MIDAS: DXSXT-only'
    elif dxsxs > 0 and dxsxt > 0:
        mid_lab = 'βI MIDAS: mixed DXSXS/DXSXT'
    else:
        mid_lab = 'βI MIDAS: none detected'

    if np.isnan(st_mean):
        flank_lab = 'NPxF flanks: NA'
    else:
        if st_mean >= 3.0:
            flank_lab = f'NPxF flanks: S/T-rich (mean={st_mean:.2f} in ±6)'
        elif st_mean >= 1.0:
            flank_lab = f'NPxF flanks: moderate S/T (mean={st_mean:.2f} in ±6)'
        else:
            flank_lab = f'NPxF flanks: low S/T (mean={st_mean:.2f} in ±6)'

    label = f'{tail_lab} | {mid_lab}'
    reason = f'NPxY={npxy}, NPxF={npxf}, DXSXS={dxsxs}, DXSXT={dxsxt}; {flank_lab}'
    return BetaClass(label=label, reason=reason)


# -----------------------------
# Orchestrator
# -----------------------------

def run_pipeline(mixed_fasta: Optional[str], alpha_fasta: Optional[str], beta_fasta: Optional[str], outdir: str,
                 groups_path: Optional[str], alpha_map: Optional[str],
                 split_mode: str = 'auto',
                 alpha_relaxed: bool = True,
                 alpha_excel: bool = False,
                 keep_unknown: bool = True,
                 alpha_script: Optional[str] = None,
                 beta_script: Optional[str] = None,
                 write_routing_debug: bool = True):

    os.makedirs(outdir, exist_ok=True)
    out_alpha = os.path.join(outdir, 'alpha')
    out_beta  = os.path.join(outdir, 'beta')
    out_merge = os.path.join(outdir, 'merged')
    os.makedirs(out_alpha, exist_ok=True)
    os.makedirs(out_beta, exist_ok=True)
    os.makedirs(out_merge, exist_ok=True)

    if mixed_fasta and (alpha_fasta or beta_fasta):
        raise ValueError('Use either --fasta (mixed) OR --alpha-fasta/--beta-fasta (separate), not both.')

    routing_rows = []
    pairing_rows = []

    if mixed_fasta:
        aligned = read_msa_auto(mixed_fasta)
        raw = ungap_upper(aligned)

        alpha_seqs = {}
        beta_seqs = {}
        unknown = {}

        for sid, seq in raw.items():
            call, ev = detect_subunit(seq)
            if split_mode == 'alpha':
                call = 'alpha'
            elif split_mode == 'beta':
                call = 'beta'

            ev_row = {'seq_id': sid, 'length': len(seq), 'call': call}
            ev_row.update(ev)
            routing_rows.append(ev_row)

            # Pairing-likelihood (computed for all sequences; you can filter to alpha later)
            tm_end, tm_found = last_tm_end(seq)
            tail_start, tail_seq, _ = beta_tail_region(seq)
            sp_flag = has_signal_peptide(seq)
            fggap_ct = count_fggap(seq)
            gff_label = find_tm_proximal_gff(tail_seq)
            pairing_rows.append({
                'Protein': sid,
                'TM': '✔' if tm_found else '❌',
                'SP': '✔' if sp_flag else '❌',
                'GFF motif': gff_label if gff_label else '❌',
                'FG-GAP': '✔' if fggap_ct >= 2 else 'weak',
                'FG-GAP_count': int(fggap_ct),
                'Pairing ability': pairing_ability_call(tm_found, sp_flag, bool(gff_label), fggap_ct),
                'Pairing_score': pairing_score(tm_found, sp_flag, bool(gff_label), fggap_ct),
            })

            if call == 'alpha':
                alpha_seqs[sid] = aligned[sid]
            elif call == 'beta':
                beta_seqs[sid] = aligned[sid]
            else:
                unknown[sid] = aligned[sid]

        if write_routing_debug and routing_rows:
            pd.DataFrame(routing_rows).to_csv(os.path.join(out_merge, 'routing_debug.csv'), index=False)

        if pairing_rows:
            pair_df = pd.DataFrame(pairing_rows)
            # order by pairing evidence (HIGH > MEDIUM > LOW); if Pairing_score exists, use it too
            pair_df['Pairing_rank'] = pair_df['Pairing ability'].map({'🟢 HIGH': 2, '🟡 MEDIUM': 1, '🔴 LOW': 0}).fillna(0)
            if 'Pairing_score' in pair_df.columns:
                pair_df = pair_df.sort_values(['Pairing_score', 'Pairing_rank', 'Protein'], ascending=[False, False, True])
            else:
                pair_df = pair_df.sort_values(['Pairing_rank', 'Protein'], ascending=[False, True])
            pair_df = pair_df.drop(columns=['Pairing_rank'])
            pair_df.to_csv(os.path.join(out_merge, 'pairing_ability.csv'), index=False)

        if alpha_seqs:
            alpha_fasta = os.path.join(out_merge, '_alpha_input.fasta')
            write_fasta(alpha_seqs, alpha_fasta)
        if beta_seqs:
            beta_fasta = os.path.join(out_merge, '_beta_input.fasta')
            write_fasta(beta_seqs, beta_fasta)
        if keep_unknown and unknown:
            write_fasta(unknown, os.path.join(out_merge, 'unknown_subunit_sequences.fasta'))

    # Locate pipeline scripts
    here = os.path.dirname(__file__)
    alpha_script = alpha_script or os.path.join(here, 'integrin_alpha_scan_and_classify.py')
    beta_script  = beta_script  or os.path.join(here, 'cobalt_integrin_beta_panels.py')

    missing = [p for p in (alpha_script, beta_script) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            'Required pipeline script(s) not found: ' + ', '.join(missing) +
            '\nFix: copy these files into the same directory as this merged script, OR pass explicit paths:\n'
            '  --alpha-script /path/to/integrin_alpha_scan_and_classify.py --beta-script /path/to/cobalt_integrin_beta_panels.py\n'
        )

    alpha_mod = import_from_path('integrin_alpha_scan_and_classify', alpha_script)
    beta_mod  = import_from_path('cobalt_integrin_beta_panels', beta_script)

    # Run α pipeline
    alpha_features_path = None
    if alpha_fasta:
        alpha_mod.analyze(
            in_path=alpha_fasta,
            outdir=out_alpha,
            groups_path=groups_path,
            user_map_path=alpha_map,
            use_relaxed_alpha=alpha_relaxed,
            write_excel=alpha_excel,
        )
        alpha_features_path = os.path.join(out_alpha, 'alpha_integrin_features.csv')

    # Run β pipeline
    beta_summary_path = None
    beta_tail_path = None
    if beta_fasta:
        beta_mod.analyze(beta_fasta, out_beta, groups_path)
        beta_summary_path = os.path.join(out_beta, 'seq_summary.csv')
        beta_tail_path = os.path.join(out_beta, 'beta_tail_motifs.csv')

    groups_map = load_groups(groups_path)

    # α table
    if alpha_features_path and os.path.exists(alpha_features_path):
        df_alpha = pd.read_csv(alpha_features_path)
        df_alpha['subunit'] = 'alpha'
        if 'seq_id' not in df_alpha.columns and 'ID' in df_alpha.columns:
            df_alpha = df_alpha.rename(columns={'ID': 'seq_id'})
        if 'group' not in df_alpha.columns:
            df_alpha['group'] = df_alpha['seq_id'].map(groups_map).fillna(df_alpha['seq_id'].map(heuristic_group))

        label_col = None
        for c in ['Alpha_integrin_subgroup', 'Alpha_subgroup', 'Subgroup', 'subgroup', 'Feature_subgroup']:
            if c in df_alpha.columns:
                label_col = c
                break
        reason_col = None
        for c in ['Subgroup_reason', 'Reason', 'reason', 'Classification_reason']:
            if c in df_alpha.columns:
                reason_col = c
                break

        if label_col:
            df_alpha = df_alpha.rename(columns={label_col: 'alpha_functional_group'})
        else:
            df_alpha['alpha_functional_group'] = '(not available)'
        if reason_col:
            df_alpha = df_alpha.rename(columns={reason_col: 'alpha_reason'})
        else:
            df_alpha['alpha_reason'] = ''

        keep = [c for c in ['seq_id', 'group', 'subunit', 'Length', 'alpha_functional_group', 'alpha_reason'] if c in df_alpha.columns]
        df_alpha_keep = df_alpha[keep].copy()
    else:
        df_alpha_keep = pd.DataFrame(columns=['seq_id', 'group', 'subunit', 'Length', 'alpha_functional_group', 'alpha_reason'])

    # β table
    if beta_summary_path and os.path.exists(beta_summary_path):
        df_beta = pd.read_csv(beta_summary_path)
        df_beta['subunit'] = 'beta'
        if 'group' not in df_beta.columns:
            df_beta['group'] = df_beta['seq_id'].map(groups_map).fillna(df_beta['seq_id'].map(heuristic_group))

        st_mean_map = {}
        if beta_tail_path and os.path.exists(beta_tail_path):
            df_tail = pd.read_csv(beta_tail_path)
            if 'motif_class' in df_tail.columns:
                df_tail = df_tail[df_tail['motif_class'] == 'NPxF']
            if 'st_flank_STcount' in df_tail.columns and len(df_tail):
                st_mean_map = df_tail.groupby('seq_id')['st_flank_STcount'].mean().to_dict()

        labels = []
        reasons = []
        for _, row in df_beta.iterrows():
            st_mean = st_mean_map.get(row['seq_id'], np.nan)
            bc = classify_beta_row(row, st_mean)
            labels.append(bc.label)
            reasons.append(bc.reason)

        df_beta['beta_functional_group'] = labels
        df_beta['beta_reason'] = reasons

        keep = [c for c in ['seq_id', 'group', 'subunit', 'length', 'beta_functional_group', 'beta_reason'] if c in df_beta.columns]
        df_beta_keep = df_beta[keep].copy()
    else:
        df_beta_keep = pd.DataFrame(columns=['seq_id', 'group', 'subunit', 'length', 'beta_functional_group', 'beta_reason'])

    # Merge
    dfm = pd.merge(df_alpha_keep, df_beta_keep, on=['seq_id', 'group'], how='outer', suffixes=('_alpha', '_beta'))

    if 'subunit_alpha' in dfm.columns or 'subunit_beta' in dfm.columns:
        def _combine_subunit(r):
            a = r.get('subunit_alpha')
            b = r.get('subunit_beta')
            if pd.notna(a):
                return a
            if pd.notna(b):
                return b
            return 'unknown'
        dfm['subunit'] = dfm.apply(_combine_subunit, axis=1)
        dfm = dfm.drop(columns=[c for c in ['subunit_alpha', 'subunit_beta'] if c in dfm.columns])

    merged_csv = os.path.join(out_merge, 'merged_integrin_classification.csv')
    dfm.to_csv(merged_csv, index=False)

    # Summary plot (counts)
    fig_path = os.path.join(out_merge, 'merged_summary_counts.png')
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=160)
        labs = []
        vals = []
        if 'alpha_functional_group' in dfm.columns:
            vc = dfm[dfm['subunit'] == 'alpha']['alpha_functional_group'].value_counts(dropna=False)
            for lab, v in vc.items():
                labs.append(f'alpha: {lab}')
                vals.append(int(v))
        if 'beta_functional_group' in dfm.columns:
            vc = dfm[dfm['subunit'] == 'beta']['beta_functional_group'].value_counts(dropna=False)
            for lab, v in vc.items():
                labs.append(f'beta: {lab}')
                vals.append(int(v))
        if labs:
            ax.bar(range(len(labs)), vals, color='#4c72b0')
            ax.set_xticks(range(len(labs)))
            ax.set_xticklabels(labs, rotation=35, ha='right')
            ax.set_ylabel('Sequence count')
            ax.set_title('Merged integrin functional-group counts')
            plt.tight_layout()
            fig.savefig(fig_path, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        with open(os.path.join(out_merge, 'plot_warning.txt'), 'w') as fh:
            fh.write(str(e) + '\n')

    print('[OK] Merged pipeline complete')
    print('  alpha out:', out_alpha)
    print('  beta  out:', out_beta)
    print('  merged  :', out_merge)
    print('  merged CSV:', merged_csv)


def main():
    ap = argparse.ArgumentParser(description='Merged α/β integrin scanning + functional-group classification (with pairing ability)')

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--fasta', help='Mixed full-length FASTA (α + β). Auto-routed by hallmarks.')
    src.add_argument('--alpha-fasta', help='FASTA for α only (bypasses auto-routing)')

    ap.add_argument('--beta-fasta', help='FASTA for β only (use with --alpha-fasta if desired)')
    ap.add_argument('--outdir', required=True, help='Output directory')
    ap.add_argument('--groups', help='Optional 2-col TSV: seq_id\tgroup')
    ap.add_argument('--alpha-map', help='Optional α subgroup map CSV/TSV (passed to α pipeline)')

    ap.add_argument('--alpha-script', help='Path to integrin_alpha_scan_and_classify.py (default: same folder as this script)')
    ap.add_argument('--beta-script', help='Path to cobalt_integrin_beta_panels.py (default: same folder as this script)')

    ap.add_argument('--split-mode', choices=['auto', 'alpha', 'beta'], default='auto',
                    help='When using --fasta: auto-route by hallmarks; or force all to alpha/beta')
    ap.add_argument('--no-relaxed-alpha', action='store_true', help='Disable relaxed α-tail detection in α pipeline')
    ap.add_argument('--alpha-excel', action='store_true', help='Write α Excel output (alpha_integrin_features.xlsx)')
    ap.add_argument('--drop-unknown', action='store_true', help='Do not write unknown_subunit_sequences.fasta')
    ap.add_argument('--no-routing-debug', action='store_true', help='Do not write merged/routing_debug.csv')

    args = ap.parse_args()

    run_pipeline(
        mixed_fasta=args.fasta,
        alpha_fasta=args.alpha_fasta,
        beta_fasta=args.beta_fasta,
        outdir=args.outdir,
        groups_path=args.groups,
        alpha_map=args.alpha_map,
        split_mode=args.split_mode,
        alpha_relaxed=(not args.no_relaxed_alpha),
        alpha_excel=args.alpha_excel,
        keep_unknown=(not args.drop_unknown),
        alpha_script=args.alpha_script,
        beta_script=args.beta_script,
        write_routing_debug=(not args.no_routing_debug),
    )


if __name__ == '__main__':
    main()
