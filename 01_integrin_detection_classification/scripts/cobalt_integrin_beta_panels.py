#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# cobalt_integrin_panels.py
#
# Build β-tail motif panels (NPxY vs. NPxF + S/T proximity) and βI MIDAS panels
# (DXSXS vs. DXSXT + minimalist 5-mer logos) from a COBALT MSA.
#
# Input  : a multiple alignment (NEXUS interleaved or FASTA). COBALT web output
#          can be downloaded as NEXUS. Optionally a two-column groups file
#          (seq_id	group) to map sequences to "human" or "parasite" (or any two labels).
# Output : CSVs and PNG figures in the chosen outdir.
#
# Usage examples:
#   python cobalt_integrin_panels.py #       --msa UM0JB17F212-alignment.txt #       --outdir results_cobalt #       --groups groups.tsv
#
#   # If you do not provide --groups, the script will try simple heuristics
#   # (IDs containing 'HUMAN' or starting with 'sp|' -> human; everything else -> parasite).
#
# Notes:
#   • β-tail is approximated as the region after 'HDR' or, if absent, after the
#     last hydrophobic 19-aa window (≥17 hydrophobic residues).
#   • NPxY/NPxF scanning is on the tail only; S/T proximity is the count of S/T
#     within ±6 aa flanking each NPxF (excluding the 4-aa motif itself).
#   • MIDAS 5-mers are collected across the full sequence where positions 1 and 3
#     are D and S, and position 5 is S/T/other (classified as DXSXS, DXSXT, or DXSX*).

import argparse
import os
import re
from collections import defaultdict, Counter
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HYDRO = set('AILVFWYMGTC')
AA = list('ACDEFGHIKLMNPQRSTVWY')
COLOR = {
    'D':'#1f77b4','E':'#1f77b4',  # acidic
    'K':'#ff7f0e','R':'#ff7f0e','H':'#ff7f0e',  # basic
    'S':'#2ca02c','T':'#2ca02c','Y':'#2ca02c','N':'#2ca02c','Q':'#2ca02c',  # polar
    'A':'#8c564b','V':'#8c564b','I':'#8c564b','L':'#8c564b','M':'#8c564b','F':'#8c564b','W':'#8c564b','P':'#8c564b','G':'#8c564b','C':'#8c564b'  # hydrophobic/other
}

# ----------------------------- I/O parsers ------------------------------------

def read_fasta(path: str) -> Dict[str,str]:
    seqs = {}
    name = None
    with open(path, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                name = line[1:].strip().split()[0]
                if name in seqs:
                    raise ValueError(f"Duplicate FASTA ID: {name}")
                seqs[name] = ''
            else:
                if name is None:
                    raise ValueError('FASTA format error: sequence before header')
                seqs[name] += line
    return seqs

def read_nexus_matrix(path: str) -> dict[str, str]:
    from collections import defaultdict
    seqs = defaultdict(str)
    inside = False
    with open(path, 'r') as fh:
        for line in fh:
            line = line.rstrip()
            if not inside:
                if line.strip().upper().startswith('MATRIX'):
                    inside = True
                continue
            # end of the matrix block
            if line.strip().startswith(';'):
                break
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            sid = parts[0]
            chunk = ''.join(parts[1:])
            seqs[sid] += chunk
    return dict(seqs)

def read_msa_auto(path: str) -> Dict[str,str]:
    # minimal auto-detection for NEXUS vs FASTA
    with open(path, 'r') as fh:
        head_lines = []
        for i in range(5):
            try:
                head_lines.append(next(fh))
            except StopIteration:
                break
    head = ''.join(head_lines)
    if '#NEXUS' in head.upper():
        return read_nexus_matrix(path)
    else:
        return read_fasta(path)

# ----------------------------- Grouping ---------------------------------------

def load_groups(path: str) -> Dict[str,str]:
    m = {}
    with open(path, 'r') as fh:
        for line in fh:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.rstrip('').split('	')
            if len(parts) < 2:
                continue
            sid, grp = parts[0], parts[1]
            m[sid] = grp
    return m

def heuristic_group(sid: str) -> str:
    # Human if header contains 'HUMAN' or starts with 'sp|'; else parasite
    if 'HUMAN' in sid.upper() or sid.startswith('sp|'):
        return 'human'
    return 'parasite'

# ------------------------ Biology-specific utilities --------------------------

def ungap(seq: str) -> str:
    return seq.replace('-', '')

def tail_region(seq: str) -> Tuple[int,str]:
    # return (start_index_1based, tail_seq)
    m = re.search('HDR', seq)
    if m:
        start = m.end()
        return start, seq[start:]
    # else last hydrophobic 19-mer with >=17 hydrophobics
    best = -1
    for i in range(len(seq)-18):
        window = seq[i:i+19]
        if sum(1 for c in window if c in HYDRO) >= 17:
            best = i
    if best != -1:
        start = best + 19
        return start, seq[start:]
    start = max(0, len(seq)-200)
    return start, seq[start:]

# ----------------------------- Logos -----------------------------------------

def logo_matrix(strings: List[str]) -> np.ndarray:
    M = np.zeros((20,5), dtype=float)
    if not strings:
        return M
    idx = {a:i for i,a in enumerate(AA)}
    for s in strings:
        if len(s) != 5: continue
        for j, ch in enumerate(s):
            if ch in idx:
                M[idx[ch], j] += 1
    colsum = M.sum(axis=0, keepdims=True)
    colsum[colsum==0] = 1
    return M/colsum

def draw_simple_logo(ax, M: np.ndarray, title: str):
    ax.set_xlim(0,5); ax.set_ylim(0,1)
    ax.set_xticks(range(5)); ax.set_xticklabels(['1','2','3','4','5'])
    ax.set_yticks([0,0.5,1.0]); ax.set_ylabel('Freq')
    ax.set_title(title)
    for j in range(5):
        col = [(AA[i], M[i,j]) for i in range(20) if M[i,j] > 0]
        col.sort(key=lambda x: x[1])
        y = 0
        for aa, h in col:
            ax.text(j+0.05, y, aa, fontsize=12, color=COLOR.get(aa,'black'),
                    va='bottom', ha='left', fontweight='bold')
            y += h
    ax.set_xlim(-0.1,5.1)

# ----------------------------- Main analysis ---------------------------------

def analyze(msa_path: str, outdir: str, groups_path: str=None):
    os.makedirs(outdir, exist_ok=True)

    aligned = read_msa_auto(msa_path)
    # ungap to raw sequences (COBALT MSA is global; motif scans are position-in-sequence based)
    raw = {sid: ungap(seq) for sid, seq in aligned.items()}

    # group mapping
    if groups_path:
        gmap = load_groups(groups_path)
    else:
        gmap = {sid: heuristic_group(sid) for sid in raw.keys()}
    # groups in deterministic order
    groups = sorted(set(gmap.values()))

    # Containers
    tail_records = []  # seq_id, group, class, motif, abs_pos, st_flank
    np_counts = {g: Counter() for g in groups}
    st_flanks = {g: [] for g in groups}

    mid5 = {g: [] for g in groups}
    mid_classes = {g: Counter() for g in groups}

    # Per-seq summary rows
    summary_rows = []

    for sid, seq in raw.items():
        grp = gmap.get(sid, 'group')
        # tail
        tstart, tail = tail_region(seq)
        npxy = 0; npxf = 0
        for m in re.finditer(r'NP.Y', tail):
            npxy += 1
            tail_records.append([sid, grp, 'NPxY', m.group(), tstart + m.start() + 1, None])
            np_counts[grp]['NPxY'] += 1
        for m in re.finditer(r'NP.F', tail):
            npxf += 1
            pos0 = m.start()
            L = max(0, pos0-6); R = min(len(tail), pos0+4+6)
            flank = tail[L:pos0] + tail[pos0+4:R]
            st_ct = sum(1 for c in flank if c in 'ST')
            st_flanks[grp].append(st_ct)
            tail_records.append([sid, grp, 'NPxF', m.group(), tstart + m.start() + 1, st_ct])
            np_counts[grp]['NPxF'] += 1
        # MIDAS 5-mers on full sequence
        dxsxs = 0; dxsxt = 0
        for i in range(len(seq)-4):
            five = seq[i:i+5]
            if five[0]=='D' and five[2]=='S':
                if five[4]=='S':
                    cls = 'DXSXS'; dxsxs += 1
                elif five[4]=='T':
                    cls = 'DXSXT'; dxsxt += 1
                else:
                    cls = 'DXSX*'
                mid5[grp].append(five)
                mid_classes[grp][cls] += 1
        summary_rows.append([sid, grp, len(seq), tstart, npxy, npxf, dxsxs, dxsxt])

    # Save CSVs
    df_tail = pd.DataFrame(tail_records, columns=[
        'seq_id','group','motif_class','motif','abs_pos_1based','st_flank_STcount']
    )
    df_tail.to_csv(os.path.join(outdir, 'beta_tail_motifs.csv'), index=False)

    summary = pd.DataFrame(summary_rows, columns=[
        'seq_id','group','length','tail_start_1based','NPxY_count_tail','NPxF_count_tail','DXSXS_count','DXSXT_count']
    )
    summary.to_csv(os.path.join(outdir, 'seq_summary.csv'), index=False)

    # ----------------- Figure 1: β-tail PTB motifs + S/T histogram -------------
    fig, axes = plt.subplots(1,2, figsize=(9,4), dpi=160)
    x = np.arange(len(groups)); width = 0.35
    y_counts = [np_counts[g]['NPxY'] for g in groups]
    f_counts = [np_counts[g]['NPxF'] for g in groups]
    axes[0].bar(x - width/2, y_counts, width, label='NPxY', color='#1f77b4')
    axes[0].bar(x + width/2, f_counts, width, label='NPxF', color='#ff7f0e')
    axes[0].set_xticks(x); axes[0].set_xticklabels(groups)
    axes[0].set_ylabel('Motif count (tails)')
    axes[0].set_title('β-tail PTB motifs (COBALT MSA)')
    axes[0].legend(frameon=False)

    bins = np.arange(-0.5, 8.5, 1.0)
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#8c564b']
    for idx_g, g in enumerate(groups):
        axes[1].hist(st_flanks[g], bins=bins, alpha=0.75, label=f'{g} NPxF', color=colors[idx_g % len(colors)])
    axes[1].set_xticks(range(0,9))
    axes[1].set_xlabel('S/T count in ±6 aa flanks (excl. motif)')
    axes[1].set_ylabel('Instances')
    axes[1].set_title('S/T proximity around NPxF')
    axes[1].legend(frameon=False)

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, 'beta_tail_panels.png'), bbox_inches='tight')
    plt.close(fig)

    # ----------------- Figure 2: βI MIDAS composition + logos ------------------
    fig2 = plt.figure(figsize=(10,5), dpi=160)
    ax1 = plt.subplot2grid((2,3),(0,0))
    ax2 = plt.subplot2grid((2,3),(0,1))
    ax3 = plt.subplot2grid((2,3),(0,2))
    ax4 = plt.subplot2grid((2,3),(1,0), colspan=3)

    classes = ['DXSXS','DXSXT','DXSX*']
    # first two groups (or replicate)
    gA = groups[0] if groups else None
    gB = groups[1] if len(groups)>1 else None

    ax1.bar(range(3), [ (mid_classes[gA][c] if gA else 0) for c in classes], color=['#2ca02c','#ff7f0e','#8c564b'])
    ax1.set_xticks(range(3)); ax1.set_xticklabels(classes, rotation=25)
    ax1.set_ylabel('Count'); ax1.set_title(f'{gA or "group1"} β — MIDAS 5-mer')

    if gB:
        ax2.bar(range(3), [mid_classes[gB][c] for c in classes], color=['#2ca02c','#ff7f0e','#8c564b'])
        ax2.set_xticks(range(3)); ax2.set_xticklabels(classes, rotation=25)
        ax2.set_ylabel('Count'); ax2.set_title(f'{gB} β — MIDAS 5-mer')
    else:
        ax2.axis('off')

    dxsxs_A = mid_classes[gA]['DXSXS'] if gA else 0
    dxsxt_A = mid_classes[gA]['DXSXT'] if gA else 0
    dxsxs_B = mid_classes[gB]['DXSXS'] if gB else 0
    dxsxt_B = mid_classes[gB]['DXSXT'] if gB else 0

    ax3.bar([0,1], [dxsxs_A, dxsxs_B], color='#2ca02c', label='DXSXS')
    ax3.bar([0,1], [dxsxt_A, dxsxt_B], bottom=[dxsxs_A, dxsxs_B], color='#ff7f0e', label='DXSXT')
    ax3.set_xticks([0,1]); ax3.set_xticklabels([gA or 'group1', gB or 'group2'])
    ax3.set_ylabel('Count'); ax3.set_title('MIDAS composition')
    ax3.legend(frameon=False)

    # logos
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axh = inset_axes(ax4, width='48%', height='80%', loc='upper left', borderpad=1)
    axp = inset_axes(ax4, width='48%', height='80%', loc='upper right', borderpad=1)
    M_h = logo_matrix(mid5[gA]) if gA else np.zeros((20,5))
    M_p = logo_matrix(mid5[gB]) if gB else np.zeros((20,5))
    draw_simple_logo(axh, M_h, f'{gA or "group1"} β — MIDAS logo')
    if gB:
        draw_simple_logo(axp, M_p, f'{gB} β — MIDAS logo')
    else:
        axp.axis('off')

    fig2.tight_layout()
    fig2.savefig(os.path.join(outdir, 'betaI_MIDAS_panels.png'), bbox_inches='tight')
    plt.close(fig2)

    print(f"Wrote outputs to: {outdir}")


def main():
    ap = argparse.ArgumentParser(description='Build β-tail PTB motif and βI MIDAS panels from a COBALT alignment.')
    ap.add_argument('--msa', required=True, help='MSA file (NEXUS interleaved from COBALT, or FASTA).')
    ap.add_argument('--outdir', required=True, help='Output directory for CSVs and PNGs.')
    ap.add_argument('--groups', help='Optional two-column TSV: seq_id	group (e.g., human/parasite).')
    args = ap.parse_args()
    analyze(args.msa, args.outdir, args.groups)

if __name__ == '__main__':
    main()
