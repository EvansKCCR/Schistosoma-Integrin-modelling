# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""rank_integrin_binding_v3_hotspots.py

Rank RGD–integrin binder models by composite biophysical metrics.

Features
--------
- Contacts (heavy-atom) within a cutoff
- Buried Surface Area (BSA): FreeSASA if available, else proxy from contacts
- Salt bridges (RGD positive vs receptor acidic)
- H-bonds (distance-only heuristic)
- MIDAS integrity heuristic around Mg2+

Additions
---------
- RGD–receptor residue contact maps (CSV + optional PNG heatmap)
- Penalized ranking when *no ARG salt bridge* is detected
- PDB sanitizer that rewrites element column (cols 77–78) from atom name, writing *_fixed.pdb
- Optional suppression of MDAnalysis INFO chatter (guessers / topology)

Notes
-----
- Many PDBs omit/garble the element field; sanitizer fixes the fixed-width element column.
- MDAnalysis guess_TopologyAttrs(...): use force_guess to overwrite existing attributes.
"""

import os
import re
import glob
import argparse
import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import MDAnalysis as mda
from MDAnalysis.core.groups import AtomGroup, Atom
from MDAnalysis.lib.distances import distance_array, capped_distance
from MDAnalysis.coordinates.PDB import PDBWriter

# Matplotlib is used only if --plots is enabled
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional FreeSASA
try:
    import freesasa  # noqa: F401
    HAVE_FREESASA = True
except Exception:
    HAVE_FREESASA = False


# ------------------------------------------------------------
# DEFAULT CONSTANTS (overridable via CLI)
# ------------------------------------------------------------
CONTACT_CUTOFF = 4.0     # Å
SALT_CUTOFF = 4.0        # Å
HBOND_CUTOFF = 3.5       # Å (distance-only heuristic)

# Mg–O windows (Å)
MG_OPTIMAL = (2.0, 2.3)
MG_ACCEPTABLE = (2.3, 2.6)

# Score weights (overridable by --weights)
W_BSA = 0.01
W_SALT = 2.0
W_HBOND = 1.5
W_CONTACTS = 0.1
W_MIDAS = 3.0

# Penalty applied if no ARG salt bridge is detected (ARG in the RGD peptide)
ARG0_PENALTY = 5.0


# ------------------------------------------------------------
# Utility / parsing helpers
# ------------------------------------------------------------

def parse_weights(s: Optional[str]) -> None:
    """Parse weight string like 'bsa=0.01,salt=2,hbond=1.5,contacts=0.1,midas=3' and update globals."""
    if not s:
        return

    mapping = {
        "bsa": "W_BSA",
        "salt": "W_SALT",
        "hbond": "W_HBOND",
        "hbonds": "W_HBOND",
        "contacts": "W_CONTACTS",
        "midas": "W_MIDAS",
    }

    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            logging.warning(f"Ignoring malformed weight token: '{p}'")
            continue
        key, val = [x.strip().lower() for x in p.split("=", 1)]
        if key not in mapping:
            logging.warning(f"Ignoring unknown weight key: '{p}'")
            continue
        try:
            fval = float(val)
        except ValueError:
            logging.warning(f"Ignoring non-numeric weight value: '{p}'")
            continue
        globals()[mapping[key]] = fval


def select_chains(u: mda.Universe, chains: List[str]) -> AtomGroup:
    """Return atoms belonging to any of the given chains.

    Tries chainID first; falls back to segid.
    """
    if not chains:
        return u.atoms[:0]

    expr = " or ".join([f"chainID {c}" for c in chains])
    ag = u.select_atoms(expr)
    if len(ag) == 0:
        expr = " or ".join([f"segid {c}" for c in chains])
        ag = u.select_atoms(expr)
    return ag


# ------------------------------------------------------------
# PDB sanitizer: rewrite element column (77-78) from atom name
# ------------------------------------------------------------

_STD_AA = {
    'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL',
    'SEC','PYL'
}

_TWO_LETTER_ELEMENTS = {
    'CL','BR','NA','MG','ZN','FE','CA','MN','CU','CO','NI','SE','CD','HG',
    'SR','CS','RB','LI','AL','SI','SN','PB','AU','AG'
}


def _infer_element_from_atom_name(atom_name: str, resname: str = '') -> str:
    """Infer element symbol from PDB atom name (cols 13-16).

    Heuristics:
    - Strip leading digits (e.g., 1HG1 -> HG1)
    - If atom name is 'CA' and residue is standard AA, treat as carbon ('C')
    - Prefer known two-letter elements when present (CL, BR, NA, MG, ...)
    - Else use first alphabetic letter
    """
    name = (atom_name or '').strip()
    rname = (resname or '').strip().upper()

    if not name:
        return ''

    name2 = re.sub(r'^\d+', '', name.upper())

    if name2 == 'CA' and rname in _STD_AA:
        return 'C'

    if len(name2) >= 2 and name2[:2] in _TWO_LETTER_ELEMENTS:
        return name2[:2]

    ch = name2[0]
    return ch if ch.isalpha() else ''


def sanitize_pdb_elements(in_pdb: str, out_pdb: Optional[str] = None) -> str:
    """Rewrite ATOM/HETATM element column (77-78) based on atom name and write *_fixed.pdb.

    If out_pdb is not provided, writes alongside input as <basename>_fixed.pdb.
    Returns output path.
    """
    if out_pdb is None:
        root, _ = os.path.splitext(in_pdb)
        out_pdb = root + '_fixed.pdb'

    with open(in_pdb, 'r', encoding='utf-8', errors='replace') as fin, \
         open(out_pdb, 'w', encoding='utf-8') as fout:
        for line in fin:
            if line.startswith(('ATOM', 'HETATM')):
                atom_name = line[12:16]
                resname = line[17:20]
                elem = _infer_element_from_atom_name(atom_name, resname)

                s = line.rstrip('\n')
                if len(s) < 80:
                    s = s.ljust(80)

                elem2 = (elem or '').rjust(2)
                s = s[:76] + elem2[:2] + s[78:]
                fout.write(s + '\n')
            else:
                fout.write(line)

    return out_pdb


# ------------------------------------------------------------
# Metric computations
# ------------------------------------------------------------

def compute_contacts(
    rgd: AtomGroup,
    receptor: AtomGroup,
    cutoff: float = CONTACT_CUTOFF,
    heavy_only: bool = True,
    fast: bool = False,
) -> int:
    """Count atom–atom contacts within cutoff (Å)."""
    if heavy_only:
        rgd = rgd.select_atoms("not name H*")
        receptor = receptor.select_atoms("not name H*")

    if len(rgd) == 0 or len(receptor) == 0:
        return 0

    if fast:
        pairs = capped_distance(rgd.positions, receptor.positions, cutoff, return_distances=False)
        return int(len(pairs))

    d = distance_array(rgd.positions, receptor.positions)
    return int(np.count_nonzero(d < cutoff))


# ---------------- Contact maps ----------------

def _format_res_label(res) -> str:
    """Consistent residue label for contact maps."""
    chain = None
    try:
        chain = res.atoms.chainIDs[0]
        if not chain:
            chain = None
    except Exception:
        chain = None

    if chain is None:
        try:
            chain = res.segid
        except Exception:
            chain = ""

    return f"{res.resname}{res.resid}:{chain}" if chain else f"{res.resname}{res.resid}"


def compute_rgd_contact_map(
    rgd: AtomGroup,
    receptor: AtomGroup,
    cutoff: float = CONTACT_CUTOFF,
    heavy_only: bool = True,
) -> pd.DataFrame:
    """Build an RGD–receptor residue-level contact map."""
    if heavy_only:
        rgd_atoms = rgd.select_atoms("not name H*")
        rec_atoms = receptor.select_atoms("not name H*")
    else:
        rgd_atoms = rgd
        rec_atoms = receptor

    if len(rgd_atoms) == 0 or len(rec_atoms) == 0:
        return pd.DataFrame()

    pairs = capped_distance(rgd_atoms.positions, rec_atoms.positions, cutoff, return_distances=False)

    rgd_res = list(rgd_atoms.residues)
    rec_res = list(rec_atoms.residues)
    idx = [_format_res_label(r) for r in rgd_res]
    cols = [_format_res_label(r) for r in rec_res]

    if pairs is None or len(pairs) == 0:
        return pd.DataFrame(0, index=idx, columns=cols, dtype=int)

    rgd_map = {r.resindex: i for i, r in enumerate(rgd_res)}
    rec_map = {r.resindex: j for j, r in enumerate(rec_res)}

    mat = np.zeros((len(rgd_res), len(rec_res)), dtype=int)
    rgd_resindices = rgd_atoms.resindices
    rec_resindices = rec_atoms.resindices

    for i_atom, j_atom in pairs:
        ri = rgd_map.get(int(rgd_resindices[int(i_atom)]))
        cj = rec_map.get(int(rec_resindices[int(j_atom)]))
        if ri is not None and cj is not None:
            mat[ri, cj] += 1

    return pd.DataFrame(mat, index=idx, columns=cols, dtype=int)


def save_contact_map(
    df_map: pd.DataFrame,
    csv_path: str,
    png_path: Optional[str] = None,
    title: str = "RGD-receptor contact map",
    binary: bool = False,
    vmin: float = 0.0,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
) -> None:
    """Save contact map to CSV and optionally a PNG heatmap.

    If binary=True, values are saved/plotted as 0/1.
    If binary=False and values lie in [0,1], the map is treated as a frequency map.
    Otherwise it is treated as a count map.
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    if df_map is None or not isinstance(df_map, pd.DataFrame) or df_map.size == 0:
        pd.DataFrame().to_csv(csv_path)
        return

    out = (df_map > 0).astype(int) if binary else df_map
    out.to_csv(csv_path)

    if not png_path or out.size == 0:
        return

    arr = out.values.astype(float)

    # Auto-scale
    if vmax is None:
        if binary:
            vmax = 1.0
        else:
            finite = arr[np.isfinite(arr)]
            if finite.size and finite.min() >= 0.0 and finite.max() <= 1.0:
                vmax = 1.0
            else:
                vmax = float(finite.max()) if finite.size else 1.0

    # Colorbar label
    if binary:
        cbar_label = "Contact (0/1)"
    else:
        finite = arr[np.isfinite(arr)]
        if finite.size and finite.min() >= 0.0 and finite.max() <= 1.0:
            cbar_label = "Contact frequency"
        else:
            cbar_label = "Contact count"

    # Figure sizing
    fig_w = max(8.0, 0.30 * out.shape[1] + 2.0)
    fig_h = max(3.0, 0.30 * out.shape[0] + 2.0)
    plt.figure(figsize=(fig_w, fig_h))

    im = plt.imshow(arr, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=cbar_label, fraction=0.046, pad=0.04)

    plt.yticks(range(out.shape[0]), out.index.tolist(), fontsize=7)

    ncols = out.shape[1]
    step = max(1, ncols // 35)
    xticks = list(range(0, ncols, step))
    plt.xticks(xticks, [out.columns[i] for i in xticks], rotation=90, fontsize=7)

    plt.xlabel("Receptor residue", fontsize=9)
    plt.ylabel("RGD residue", fontsize=9)
    plt.title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()



# ---------------- Hotspot maps (non-zero contacts) ----------------

def compute_contact_hotspot_map(df_map: pd.DataFrame, binary: bool = True, threshold: float = 0.0) -> pd.DataFrame:
    """Return a contact hotspot map keeping only non-zero contacts.

    Parameters
    ----------
    df_map : pd.DataFrame
        Contact matrix (counts, binary, or frequency).
    binary : bool
        If True, hotspot map is 0/1 presence; if False, retains original values.
    threshold : float
        Minimum value to be considered a contact (default > 0).
    """
    if df_map is None or not isinstance(df_map, pd.DataFrame) or df_map.size == 0:
        return pd.DataFrame() if df_map is None else df_map
    mask = df_map > float(threshold)
    hot = mask.astype(int) if binary else df_map.where(mask, 0)
    # drop fully-zero rows/cols to focus on hotspots
    if hot.shape[0] > 0:
        hot = hot.loc[hot.sum(axis=1) > 0]
    if hot.shape[1] > 0:
        hot = hot.loc[:, hot.sum(axis=0) > 0]
    return hot


def write_hotspot_pairs(df_map: pd.DataFrame, csv_path: str, min_value: float = 0.0) -> None:
    """Write a long-format list of non-zero hotspot pairs sorted by value."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    if df_map is None or df_map.empty:
        pd.DataFrame(columns=["rgd_res", "receptor_res", "value"]).to_csv(csv_path, index=False)
        return
    long = df_map.stack().reset_index()
    long.columns = ["rgd_res", "receptor_res", "value"]
    long = long[long["value"] > float(min_value)].sort_values("value", ascending=False)
    long.to_csv(csv_path, index=False)


def save_hotspot_summaries(df_map: pd.DataFrame, outdir: str, tag: str) -> None:
    """Save per-residue hotspot summaries for RGD and receptor."""
    if df_map is None or df_map.empty:
        return
    os.makedirs(outdir, exist_ok=True)
    rgd_hot = df_map.sum(axis=1).sort_values(ascending=False).rename("hotspot")
    rec_hot = df_map.sum(axis=0).sort_values(ascending=False).rename("hotspot")
    rgd_hot.to_csv(os.path.join(outdir, f"hotspot_rgd_{tag}.csv"))
    rec_hot.to_csv(os.path.join(outdir, f"hotspot_receptor_{tag}.csv"))

# ---------------- Ensemble contact frequency (across models) ----------------

DEFAULT_LIGAND_PATTERNS: Dict[str, List[str]] = {
    "GRGDSP": ["rgd_", "rgd0", "rgd1", "rgd2", "rgd3", "grgdsp"],
    "GRGESP": ["mutated_rgd", "grgesp"],
    "VP7_nonRGD": ["non_rgd", "vp7", "newlcnpdm"],
    "RGD4C": ["rgd4c"],
    "iRGD": ["irgd"],
    "GYRGDGQ": ["gyrgdgq"],
    "IKVAV": ["ikvav"],
    "YIGSR": ["yigsr"],
    "GFOGER": ["gfoger", "gfoqer"],
}


def get_ligand_class(model_name: str, patterns: Optional[Dict[str, List[str]]] = None) -> str:
    """Assign ligand class from model filename using substring patterns."""
    patterns = patterns or DEFAULT_LIGAND_PATTERNS
    nm = str(model_name).lower()
    for lig, keys in patterns.items():
        for k in keys:
            if k.lower() in nm:
                return lig
    return "Other"


def aggregate_contact_frequency(contact_maps_binary: List[pd.DataFrame]) -> Tuple[pd.DataFrame, int]:
    """Aggregate per-model binary contact maps into a frequency map in [0,1]."""
    if not contact_maps_binary:
        return pd.DataFrame(), 0

    all_rows = sorted(set().union(*[m.index for m in contact_maps_binary]))
    all_cols = sorted(set().union(*[m.columns for m in contact_maps_binary]))

    aligned = [m.reindex(index=all_rows, columns=all_cols, fill_value=0) for m in contact_maps_binary]
    stacked = np.stack([m.values for m in aligned], axis=0)
    freq = stacked.mean(axis=0)

    return pd.DataFrame(freq, index=all_rows, columns=all_cols), len(contact_maps_binary)


def compute_frequency_hotspots(freq_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Keep only cells with freq > threshold; drop empty rows/cols."""
    if freq_df is None or freq_df.empty:
        return pd.DataFrame()
    hot = freq_df.where(freq_df > float(threshold), other=0.0)
    if hot.shape[0] > 0:
        hot = hot.loc[hot.sum(axis=1) > 0]
    if hot.shape[1] > 0:
        hot = hot.loc[:, hot.sum(axis=0) > 0]
    return hot


def plot_contact_frequency_grid(
    freq_maps: Dict[str, pd.DataFrame],
    n_models: Dict[str, int],
    out_png: str,
    threshold: float = 0.5,
    vmax: float = 1.0,
    cmap: str = "viridis",
    ncols: int = 3,
    figsize_per_panel: Tuple[float, float] = (4.2, 3.8),
) -> None:
    """Plot multiple ligand-class frequency hotspot maps in one grid figure."""
    ligands = sorted([k for k, v in freq_maps.items() if v is not None and not v.empty])
    if not ligands:
        return

    all_rows = sorted(set().union(*[freq_maps[lig].index for lig in ligands]))
    all_cols = sorted(set().union(*[freq_maps[lig].columns for lig in ligands]))
    aligned = {lig: freq_maps[lig].reindex(index=all_rows, columns=all_cols, fill_value=0.0) for lig in ligands}

    n_panels = len(ligands)
    nrows = int(math.ceil(n_panels / float(ncols)))

    fig_w = figsize_per_panel[0] * ncols
    fig_h = figsize_per_panel[1] * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False)

    last_im = None
    for i, lig in enumerate(ligands):
        r = i // ncols
        c = i % ncols
        ax = axes[r][c]
        mat = aligned[lig].values.astype(float)
        last_im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto", interpolation="nearest")

        ax.set_xticks(range(len(all_cols)))
        ax.set_xticklabels(all_cols, rotation=90, fontsize=6)
        ax.set_yticks(range(len(all_rows)))
        ax.set_yticklabels(all_rows, fontsize=6)

        ax.set_xlabel("Receptor residue", fontsize=8)
        ax.set_ylabel("RGD residue", fontsize=8)
        ax.set_title(f"{lig} (n={n_models.get(lig, 0)})", fontsize=10)

    for j in range(n_panels, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r][c].axis("off")

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
        cbar.set_label("Contact frequency", fontsize=10)

    fig.suptitle(f"RGD-Sm\u03b11/Sm\u03b21 contact frequency (freq>{threshold})", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


# ---------------- BSA ----------------

def compute_bsa_proxy(contact_count: int) -> float:
    return float(contact_count) * 6.0


def compute_bsa_freesasa(rgd: AtomGroup, receptor: AtomGroup) -> Optional[float]:
    """Compute buried surface area using FreeSASA.

    BSA = SASA(RGD) + SASA(Receptor) - SASA(Complex)
    """
    if not HAVE_FREESASA or len(rgd) == 0 or len(receptor) == 0:
        return None

    import tempfile

    complex_pdb = rgd_pdb = rec_pdb = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f_complex, \
             tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f_rgd, \
             tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f_rec:
            complex_pdb, rgd_pdb, rec_pdb = f_complex.name, f_rgd.name, f_rec.name

        with PDBWriter(complex_pdb) as w:
            w.write(rgd | receptor)
        with PDBWriter(rgd_pdb) as w:
            w.write(rgd)
        with PDBWriter(rec_pdb) as w:
            w.write(receptor)

        sasa_complex = freesasa.calc(freesasa.Structure(complex_pdb)).totalArea()
        sasa_rgd = freesasa.calc(freesasa.Structure(rgd_pdb)).totalArea()
        sasa_rec = freesasa.calc(freesasa.Structure(rec_pdb)).totalArea()

        bsa = (sasa_rgd + sasa_rec) - sasa_complex
        return max(float(bsa), 0.0)

    except Exception as e:
        logging.debug(f"FreeSASA failed: {e}")
        return None
    finally:
        for fn in (complex_pdb, rgd_pdb, rec_pdb):
            if fn and os.path.exists(fn):
                try:
                    os.unlink(fn)
                except Exception:
                    pass


# ---------------- Salt bridges / H-bonds ----------------

def compute_salt_bridges(rgd: AtomGroup, receptor: AtomGroup, cutoff: float = SALT_CUTOFF) -> int:
    positive = rgd.select_atoms("resname ARG LYS and name NH1 NH2 NE NZ")
    negative = receptor.select_atoms("resname ASP GLU and name OD1 OD2 OE1 OE2")
    if len(positive) == 0 or len(negative) == 0:
        return 0
    d = distance_array(positive.positions, negative.positions)
    return int(np.count_nonzero(d < cutoff))


def compute_arg_salt_bridges(rgd: AtomGroup, receptor: AtomGroup, cutoff: float = SALT_CUTOFF) -> Tuple[int, int]:
    positive = rgd.select_atoms("resname ARG and name NH1 NH2 NE")
    negative = receptor.select_atoms("resname ASP GLU and name OD1 OD2 OE1 OE2")
    if len(positive) == 0 or len(negative) == 0:
        return 0, 0

    pairs = capped_distance(positive.positions, negative.positions, cutoff, return_distances=False)
    if pairs is None or len(pairs) == 0:
        return 0, 0

    neg_residx = negative.resindices
    touched = {int(neg_residx[int(j)]) for _, j in pairs}
    return int(len(pairs)), int(len(touched))


def compute_hbonds(rgd: AtomGroup, receptor: AtomGroup, cutoff: float = HBOND_CUTOFF, bidirectional: bool = False) -> int:
    donors_rgd = rgd.select_atoms("name N NE NH1 NH2 ND1 ND2 NZ")
    accept_receptor = receptor.select_atoms("name O OD1 OD2 OE1 OE2 OG OG1")

    total = 0
    if len(donors_rgd) > 0 and len(accept_receptor) > 0:
        d = distance_array(donors_rgd.positions, accept_receptor.positions)
        total += int(np.count_nonzero(d < cutoff))

    if bidirectional:
        donors_rec = receptor.select_atoms("name N NE NH1 NH2 ND1 ND2 NZ")
        accept_rgd = rgd.select_atoms("name O OD1 OD2 OE1 OE2 OG OG1")
        if len(donors_rec) > 0 and len(accept_rgd) > 0:
            d2 = distance_array(donors_rec.positions, accept_rgd.positions)
            total += int(np.count_nonzero(d2 < cutoff))

    return total


# ---------------- MIDAS / Mg2+ ----------------

def _chain_expr(chains: List[str]) -> str:
    """Build a robust selection expression for multiple chains/segids.

    Returns a string like: (chainID A or segid A) or (chainID B or segid B)
    """
    parts = [f"(chainID {c} or segid {c})" for c in chains if str(c).strip()]
    return " or ".join(parts)


def _pick_mg_atom(u: mda.Universe, rgd_chain: str, receptor_chains: List[str], mg_chain: Optional[str]) -> Optional[Atom]:
    """Pick the most plausible MIDAS Mg2+ atom.

    Strategy (robust to messy PDBs):
      1) Find Mg candidates by element/resname/name.
      2) If mg_chain is provided, prefer Mg atoms on that chain/segid.
      3) Score each candidate by:
         - proximity to ligand acidic oxygens (ASP/GLU carboxylate O atoms)
         - presence of a reasonable receptor donor environment (O/N atoms) nearby

    This improves behavior when multiple metals are present and supports RGD and RGE-like ligands.
    """
    mg_candidates = u.select_atoms("(element Mg) or (resname MG MG2) or (name MG)")
    if len(mg_candidates) == 0:
        return None

    if mg_chain:
        narrowed = u.select_atoms(
            f"(chainID {mg_chain} or segid {mg_chain}) and ((element Mg) or (resname MG MG2) or (name MG))"
        )
        if len(narrowed) > 0:
            mg_candidates = narrowed

    # Ligand acidic oxygens: Asp (OD1/OD2) and Glu (OE1/OE2)
    lig_acid = u.select_atoms(
        f"(chainID {rgd_chain} or segid {rgd_chain}) and resname ASP GLU and name OD1 OD2 OE1 OE2"
    )

    # Receptor donor pool (exclude waters); use both chainID and segid
    rec_expr = _chain_expr(receptor_chains)
    if rec_expr:
        rec_don_pool = u.select_atoms(
            f"({rec_expr}) and (name O* N*) and not resname HOH WAT"
        )
    else:
        rec_don_pool = u.atoms[:0]

    best = mg_candidates[0]
    best_score = -1e18

    donor_radius = 3.0  # Å: common inner-sphere donor definition window

    for mg in mg_candidates:
        score = 0.0

        # Reward closeness to ligand carboxylate O if present
        if len(lig_acid) > 0:
            d = np.linalg.norm(lig_acid.positions - mg.position[None, :], axis=1)
            lig_min = float(np.min(d))
            # Typical Mg–O distances ~2.1–2.3 Å; allow up to ~2.6 as acceptable
            if lig_min <= 2.3:
                score += 10.0
            elif lig_min <= 2.6:
                score += 5.0
            elif lig_min <= 3.0:
                score += 1.0
            else:
                score -= 5.0

        # Reward a reasonable donor environment on receptor
        if len(rec_don_pool) > 0:
            drec = np.linalg.norm(rec_don_pool.positions - mg.position[None, :], axis=1)
            n_don = int(np.count_nonzero(drec <= donor_radius))
            score += 2.0 * min(n_don, 6)  # saturate around expected CN
            if n_don < 3:
                score -= 4.0

        if score > best_score:
            best_score = score
            best = mg

    return best




def compute_midas_score(
    u: mda.Universe,
    rgd_chain: str,
    receptor_chains: List[str],
    mg_chain: Optional[str],
    mg_opt: Tuple[float, float],
    mg_acc: Tuple[float, float],
) -> Tuple[float, Dict[str, float]]:
    """Robust MIDAS evaluation around Mg2+.

    This function preserves legacy diagnostics keys used elsewhere in the script
    (AspMg_min_dist, Mg_coord_count, Mg_coord_var) while upgrading the logic:
      - Ligand acidic atoms include ASP/GLU carboxylate oxygens (OD1/OD2/OE1/OE2)
      - Inner-sphere donors are O/N atoms within 3.0 Å of Mg (excluding water)
      - Coordination number is measured within mg_acc[1] (default 2.6 Å)
      - Adds a simple octahedral-geometry heuristic when enough donors exist

    Returns:
      midas_score: float
      diag: dict with diagnostic fields
    """
    diag: Dict[str, float] = {
        # legacy
        "AspMg_min_dist": np.nan,
        "Mg_coord_count": 0,
        "Mg_coord_var": np.nan,
        # additional robust diagnostics
        "AcidMg_n_inner": 0,
        "Donor_count_3A": 0,
        "MgO_mean": np.nan,
        "MgO_std": np.nan,
        "Octahedral_heuristic": np.nan,
        "Ligand_denticity": 0,
    }

    mg_atom = _pick_mg_atom(u, rgd_chain, receptor_chains, mg_chain)
    if mg_atom is None:
        return 0.0, diag

    # Ligand acidic oxygens (Asp/Glu)
    lig_acid = u.select_atoms(
        f"(chainID {rgd_chain} or segid {rgd_chain}) and resname ASP GLU and name OD1 OD2 OE1 OE2"
    )

    score = 0.0

    # ---- Ligand carboxylate–Mg distance quality ----
    if len(lig_acid) > 0:
        d_acid = np.linalg.norm(lig_acid.positions - mg_atom.position[None, :], axis=1)
        acid_min = float(np.min(d_acid))
        diag["AspMg_min_dist"] = acid_min  # keep legacy column name

        # how many ligand oxygens are within the acceptable inner-sphere window
        n_inner_lig = int(np.count_nonzero(d_acid <= float(mg_acc[1])))
        diag["AcidMg_n_inner"] = n_inner_lig
        diag["Ligand_denticity"] = 2 if n_inner_lig >= 2 else (1 if n_inner_lig == 1 else 0)

        if mg_opt[0] <= acid_min <= mg_opt[1]:
            score += 5.0
        elif mg_acc[0] <= acid_min <= mg_acc[1]:
            score += 2.0
        elif acid_min > 3.0:
            score -= 3.0

    # ---- Donor environment (inner-sphere) ----
    donor_radius = 3.0
    heavy = u.select_atoms("not name H*")
    d_all = np.linalg.norm(heavy.positions - mg_atom.position[None, :], axis=1)
    near = heavy[d_all <= float(donor_radius)]

    # donors: O/N, exclude waters
    donors = near.select_atoms("(name O* N*) and not resname HOH WAT")
    diag["Donor_count_3A"] = int(len(donors))

    if len(donors) > 0:
        d_don = np.linalg.norm(donors.positions - mg_atom.position[None, :], axis=1)
        inner = d_don[d_don <= float(mg_acc[1])]
        ncoord = int(inner.size)
        diag["Mg_coord_count"] = ncoord

        if inner.size > 1:
            diag["Mg_coord_var"] = float(np.var(inner))
            diag["MgO_std"] = float(np.std(inner))
        if inner.size > 0:
            diag["MgO_mean"] = float(np.mean(inner))

        # Reward reasonable coordination number; Mg often CN ~6 (sometimes 5)
        score += float(ncoord) * 1.0
        if ncoord < 4:
            score -= 3.0
        elif ncoord > 6:
            score -= 1.0

        # Penalize spread in inner-sphere distances (tighter shells look better)
        if np.isfinite(diag["Mg_coord_var"]):
            score -= float(diag["Mg_coord_var"])

        # ---- Octahedral heuristic (0..1) ----
        if inner.size >= 4:
            vecs = donors.positions[d_don <= float(mg_acc[1])] - mg_atom.position[None, :]
            norms = np.linalg.norm(vecs, axis=1)
            vecs = vecs[norms > 1e-6] / norms[norms > 1e-6][:, None]
            if vecs.shape[0] >= 4:
                dots = np.dot(vecs, vecs.T)
                iu = np.triu_indices(dots.shape[0], k=1)
                vals = dots[iu]
                near0 = float(np.mean(np.abs(vals) < 0.25))
                near_opp = float(np.mean(vals < -0.85))
                oct_h = float(np.clip(0.7 * near0 + 0.3 * near_opp, 0.0, 1.0))
                diag["Octahedral_heuristic"] = oct_h
                score += 2.0 * oct_h

    return float(score), diag


# ------------------------------------------------------------
# Composite score
# ------------------------------------------------------------

def compute_biological_score(contacts: int, bsa: float, salt: int, hbonds: int, midas_score: float) -> float:
    return (
        W_BSA * bsa +
        W_SALT * salt +
        W_HBOND * hbonds +
        W_CONTACTS * contacts +
        W_MIDAS * midas_score
    )


# ------------------------------------------------------------
# Model analysis
# ------------------------------------------------------------

def analyze_model(
    pdb: str,
    rgd_chains: List[str],
    receptor_chains: List[str],
    mg_chain: Optional[str],
    use_freesasa: bool,
    heavy_only: bool,
    hbonds_bidirectional: bool,
    fast_contacts: bool,
    mg_opt: Tuple[float, float],
    mg_acc: Tuple[float, float],
) -> Optional[Dict[str, float]]:
    try:
        u = mda.Universe(pdb)
        # Overwrite elements (avoids "already read" non-overwrite behavior)
        try:
            u.guess_TopologyAttrs(force_guess=['elements'])
        except Exception:
            pass

        rgd = select_chains(u, rgd_chains)
        receptor = select_chains(u, receptor_chains)

        if len(rgd) == 0 or len(receptor) == 0:
            logging.warning(f"Skipping {pdb}: empty RGD or receptor selection.")
            return None

        contacts = compute_contacts(rgd, receptor, cutoff=CONTACT_CUTOFF, heavy_only=heavy_only, fast=fast_contacts)

        bsa = None
        if use_freesasa and HAVE_FREESASA:
            bsa = compute_bsa_freesasa(rgd, receptor)
        if bsa is None:
            bsa = compute_bsa_proxy(contacts)

        salt = compute_salt_bridges(rgd, receptor, cutoff=SALT_CUTOFF)
        arg_pairs, arg_res = compute_arg_salt_bridges(rgd, receptor, cutoff=SALT_CUTOFF)
        hbonds = compute_hbonds(rgd, receptor, cutoff=HBOND_CUTOFF, bidirectional=hbonds_bidirectional)

        midas_score, diag = compute_midas_score(u, rgd_chains[0], receptor_chains, mg_chain, mg_opt, mg_acc)

        bio_score = compute_biological_score(contacts, float(bsa), salt, hbonds, midas_score)

        arg0_pen = (-abs(ARG0_PENALTY) if (ARG0_PENALTY and arg_res == 0) else 0.0)
        rank_score = float(bio_score) + float(arg0_pen)
        classification = ("ARG_salt_present" if arg_res > 0 else "penalized_no_ARG_salt")

        out = {
            "model": os.path.basename(pdb),
            "contacts": int(contacts),
            "BSA": float(bsa),
            "salt_bridges": int(salt),
            "ARG_salt_pairs": int(arg_pairs),
            "ARG_salt_residues": int(arg_res),
            "ARG0_penalty": float(arg0_pen),
            "H_bonds": int(hbonds),
            "MIDAS_score": float(midas_score),
            "Biological_score": float(bio_score),
            "Rank_score": float(rank_score),
            "Classification": classification,
            "AspMg_min_dist": diag["AspMg_min_dist"],
            "Mg_coord_count": diag["Mg_coord_count"],
            "Mg_coord_var": diag["Mg_coord_var"],
            "AcidMg_n_inner": diag.get("AcidMg_n_inner", 0),
            "Donor_count_3A": diag.get("Donor_count_3A", 0),
            "MgO_mean": diag.get("MgO_mean", np.nan),
            "MgO_std": diag.get("MgO_std", np.nan),
            "Octahedral_heuristic": diag.get("Octahedral_heuristic", np.nan),
            "Ligand_denticity": diag.get("Ligand_denticity", 0),
        }
        return out

    except Exception as e:
        logging.exception(f"Skipping {pdb}: {e}")
        return None


# ------------------------------------------------------------
# Reporting & plots
# ------------------------------------------------------------

def write_report(top_row: pd.Series, path: str) -> None:
    with open(path, "w") as f:
        f.write("RGD–Integrin Ranking Report\n")
        f.write("===========================\n\n")
        f.write(f"Top model: {top_row['model']}\n\n")
        f.write("Key metrics:\n")
        f.write(f"  Biological_score       : {top_row['Biological_score']:.6f}\n")
        f.write(f"  Rank_score (penalized) : {top_row['Rank_score']:.6f}\n")
        f.write(f"  BSA (Å^2)              : {top_row['BSA']:.3f}\n")
        f.write(f"  Contacts               : {top_row['contacts']}\n")
        f.write(f"  H_bonds                : {top_row['H_bonds']}\n")
        f.write(f"  Salt_bridges           : {top_row['salt_bridges']}\n")
        f.write(f"  ARG_salt_residues      : {top_row['ARG_salt_residues']}\n")
        f.write(f"  ARG0_penalty           : {top_row['ARG0_penalty']}\n")
        f.write(f"  Classification         : {top_row['Classification']}\n")
        f.write(f"  MIDAS_score            : {top_row['MIDAS_score']:.6f}\n")
        f.write("\nMIDAS diagnostics:\n")
        f.write(f"  Asp–Mg min dist (Å)    : {top_row['AspMg_min_dist']}\n")
        f.write(f"  Mg coord count         : {top_row['Mg_coord_count']}\n")
        f.write(f"  Mg coord variance      : {top_row['Mg_coord_var']}\n")


def make_plots(df: pd.DataFrame, outdir: str, top_n: int) -> None:
    os.makedirs(outdir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.scatter(df["BSA"], df["Rank_score"], c="tab:blue", alpha=0.7)
    plt.xlabel("BSA (Å²)")
    plt.ylabel("Rank score (penalized)")
    plt.title("Rank score (penalized) vs BSA")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "score_vs_BSA.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(df["H_bonds"], df["Rank_score"], c="tab:green", alpha=0.7)
    plt.xlabel("H-bonds (count)")
    plt.ylabel("Rank score (penalized)")
    plt.title("Rank score (penalized) vs H-bonds")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "score_vs_Hbonds.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(df["contacts"], df["Rank_score"], c="tab:orange", alpha=0.7)
    plt.xlabel("Contacts (count)")
    plt.ylabel("Rank score (penalized)")
    plt.title("Rank score (penalized) vs contacts")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "score_vs_contacts.png"), dpi=200)
    plt.close()

    head = df.head(top_n) if top_n > 0 else df
    plt.figure(figsize=(max(6, 0.45 * len(head)), 4))
    plt.bar(head["model"], head["Rank_score"], color="tab:purple")
    plt.ylabel("Rank score (penalized)")
    plt.title(f"Top {len(head)} models")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "topN_scores_bar.png"), dpi=200)
    plt.close()


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Rank RGD–integrin binders (v2: contact maps + ARG0 penalty + sanitizer + quiet MDAnalysis)."
    )
    p.add_argument("--input_dir", required=True, help="Directory with PDB files.")
    p.add_argument("--rgd_chain", default="C", help="Chain/segid for RGD peptide (single or comma-separated).")
    p.add_argument("--receptor_chains", default="A,B", help="Comma-separated chains/segids for receptor.")
    p.add_argument("--mg_chain", default=None, help="Optional chain/segid hint for Mg2+ (auto-detect used regardless).")
    p.add_argument("--n_jobs", type=int, default=1, help="Parallel jobs.")
    p.add_argument("--output", default="RGD_binder_ranking.csv", help="Output CSV file.")

    p.add_argument("--use_freesasa", action="store_true", help="Use FreeSASA for BSA if available.")
    p.add_argument("--heavy_only", action="store_true", help="Ignore hydrogens for contact counts.")
    p.add_argument("--hbonds_bidirectional", action="store_true", help="Count receptor→RGD H-bonds as well.")
    p.add_argument("--fast_contacts", action="store_true", help="Faster contact counting using capped_distance.")
    p.add_argument("--freq_threshold", type=float, default=0.5,
                   help="Threshold for aggregate contact frequency heatmap (keep only values > this; e.g. 0.5).")

    p.add_argument("--log", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]) 
    p.add_argument("--weights", default=None,
                   help="Override weights, e.g. 'bsa=0.01,salt=2,hbond=1.5,contacts=0.1,midas=3'")

    p.add_argument("--top_n", type=int, default=0, help="Export top-N rows to 'RGD_topN.csv' (0=all).")
    p.add_argument("--plots", action="store_true", help="Save quick plots (PNG) to 'plots/' directory.")
    p.add_argument("--report", action="store_true", help="Write a text report for the top model.")

    # ARG0 penalty
    p.add_argument("--arg0_penalty", type=float, default=ARG0_PENALTY,
                   help="Penalty (subtract) if no ARG salt bridge is detected (0 disables).")

    # Contact maps
    p.add_argument("--contact_map", choices=["none", "top", "all"], default="top",
                   help="Generate contact map(s): none | top | all")
    p.add_argument("--contact_map_dir", default="contact_maps",
                   help="Output directory for contact map CSV/PNG.")
    p.add_argument("--contact_map_binary", action="store_true",
                   help="Save contact map as binary (0/1) instead of counts.")

    # Contact frequency (ensemble)
    p.add_argument("--contact_frequency", choices=["none", "subset", "by_ligand", "grid"], default="grid",
                   help="Generate contact frequency outputs: none | subset | by_ligand | grid")
    p.add_argument("--freq_models", choices=["all", "top"], default="all",
                   help="Which models to use for contact frequency: all models | top-ranked subset")
    p.add_argument("--freq_top_n", type=int, default=50,
                   help="If --freq_models=top, number of top-ranked models to aggregate (default 50)")
    p.add_argument("--ligand_ncols", type=int, default=3,
                   help="Number of columns for combined ligand grid figure (default 3)")
    p.add_argument("--min_models_per_ligand", type=int, default=2,
                   help="Minimum models required to plot a ligand class (default 2)")

    # Sanitizer + quiet
    p.add_argument("--sanitize_pdb", action="store_true",
                   help="Rewrite PDB element column (77-78) from atom name and analyze *_fixed.pdb files.")
    p.add_argument("--quiet_mda", action="store_true",
                   help="Suppress MDAnalysis INFO messages (guessers / topology).")

    # Geometry overrides
    p.add_argument("--contact_cutoff", type=float, default=CONTACT_CUTOFF)
    p.add_argument("--salt_cutoff", type=float, default=SALT_CUTOFF)
    p.add_argument("--hbond_cutoff", type=float, default=HBOND_CUTOFF)
    p.add_argument("--midas_opt", type=float, nargs=2, default=list(MG_OPTIMAL), metavar=("LOW", "HIGH"))
    p.add_argument("--midas_acc", type=float, nargs=2, default=list(MG_ACCEPTABLE), metavar=("LOW", "HIGH"))

    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log), format="%(levelname)s: %(message)s")

    # Quiet MDAnalysis loggers if requested
    if args.quiet_mda and getattr(logging, args.log) != logging.DEBUG:
        for lname in [
            "MDAnalysis",
            "MDAnalysis.core",
            "MDAnalysis.core.universe",
            "MDAnalysis.topology",
            "MDAnalysis.topology.PDBParser",
            "MDAnalysis.guesser",
            "MDAnalysis.guesser.base",
            "MDAnalysis.guesser.default_guesser",
        ]:
            lg = logging.getLogger(lname)
            lg.setLevel(logging.WARNING)
            lg.propagate = False

    # Override globals from CLI
    global CONTACT_CUTOFF, SALT_CUTOFF, HBOND_CUTOFF, MG_OPTIMAL, MG_ACCEPTABLE, ARG0_PENALTY
    CONTACT_CUTOFF = float(args.contact_cutoff)
    SALT_CUTOFF = float(args.salt_cutoff)
    HBOND_CUTOFF = float(args.hbond_cutoff)
    MG_OPTIMAL = tuple(args.midas_opt)
    MG_ACCEPTABLE = tuple(args.midas_acc)
    ARG0_PENALTY = float(args.arg0_penalty)

    parse_weights(args.weights)

    if not os.path.isdir(args.input_dir):
        logging.error(f"Input dir not found: {args.input_dir}")
        return

    pdb_files = sorted(glob.glob(os.path.join(args.input_dir, "*.pdb")))
    if not pdb_files:
        logging.error(f"No PDB files found in: {args.input_dir}")
        return

    if args.sanitize_pdb:
        print("Sanitizing PDB element columns -> writing *_fixed.pdb files ...")
        pdb_files = [sanitize_pdb_elements(p) for p in pdb_files]

    rgd_chains = [c.strip() for c in str(args.rgd_chain).split(",") if c.strip()]
    receptor_chains = [c.strip() for c in str(args.receptor_chains).split(",") if c.strip()]

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(analyze_model)(
            pdb,
            rgd_chains,
            receptor_chains,
            args.mg_chain,
            args.use_freesasa,
            args.heavy_only,
            args.hbonds_bidirectional,
            args.fast_contacts,
            MG_OPTIMAL,
            MG_ACCEPTABLE,
        )
        for pdb in pdb_files
    )

    results = [r for r in results if r is not None]
    if not results:
        logging.error("No valid models processed.")
        return

    df = pd.DataFrame(results).sort_values(by="Rank_score", ascending=False)
    df.to_csv(args.output, index=False)

    top = df.iloc[0]
    print("\nTop-ranked RGD binder (with ARG0 penalty applied if needed):")
    print(top.to_string())
    print("\nFull ranking saved to:", args.output)


    # Contact maps
    if args.contact_map and args.contact_map != "none":
        os.makedirs(args.contact_map_dir, exist_ok=True)
        base_to_path = {os.path.basename(p): p for p in pdb_files}

        def _load_map(model_name: str) -> Optional[pd.DataFrame]:
            pdb_path = base_to_path.get(model_name)
            if not pdb_path:
                return None
            u_cm = mda.Universe(pdb_path)
            try:
                u_cm.guess_TopologyAttrs(force_guess=['elements'])
            except Exception:
                pass
            rgd_cm = select_chains(u_cm, rgd_chains)
            rec_cm = select_chains(u_cm, receptor_chains)
            if len(rgd_cm) == 0 or len(rec_cm) == 0:
                return None
            df_map = compute_rgd_contact_map(rgd_cm, rec_cm, cutoff=CONTACT_CUTOFF, heavy_only=args.heavy_only)
            if df_map is None or df_map.empty:
                return None
            return df_map

        def _save_per_model(model_name: str, tag: str) -> Optional[pd.DataFrame]:
            df_map = _load_map(model_name)
            if df_map is None:
                logging.warning(f"Contact map skipped for: {model_name}")
                return None
            bin_map = (df_map > 0).astype(int)
            csv_path = os.path.join(args.contact_map_dir, f"contact_map_{tag}.csv")
            png_path = os.path.join(args.contact_map_dir, f"contact_map_{tag}.png") if args.plots else None
            save_contact_map(
                bin_map if args.contact_map_binary else df_map,
                csv_path=csv_path,
                png_path=png_path,
                title=f"RGD-receptor contact map ({tag})",
                binary=bool(args.contact_map_binary),
            )
            return bin_map

        # Save requested per-model maps
        if args.contact_map == "top":
            _save_per_model(str(top["model"]), "top")
        elif args.contact_map == "all":
            for model_name in df["model"].astype(str).tolist():
                tag = os.path.splitext(model_name)[0]
                _save_per_model(model_name, tag)

        # Ensemble contact frequency (correct definition)
        if args.contact_frequency and args.contact_frequency != "none":
            if args.freq_models == "all":
                freq_models = df["model"].astype(str).tolist()
            else:
                freq_models = df.head(int(args.freq_top_n))["model"].astype(str).tolist()

            bin_maps_all: List[pd.DataFrame] = []
            bin_maps_by_lig: Dict[str, List[pd.DataFrame]] = {}

            for model_name in freq_models:
                df_map = _load_map(model_name)
                if df_map is None:
                    continue
                bin_map = (df_map > 0).astype(int)
                bin_maps_all.append(bin_map)
                lig = get_ligand_class(model_name)
                bin_maps_by_lig.setdefault(lig, []).append(bin_map)

            if bin_maps_all:
                freq_df, n_ok = aggregate_contact_frequency(bin_maps_all)
                hot_freq = compute_frequency_hotspots(freq_df, threshold=float(args.freq_threshold))

                tag = f"gt{str(args.freq_threshold).replace('.', 'p')}"
                freq_csv = os.path.join(args.contact_map_dir, f"contact_frequency_{tag}.csv")
                freq_png = os.path.join(args.contact_map_dir, f"contact_frequency_{tag}.png") if args.plots else None

                # Title requested: freq>0.5, n=number of models
                title = f"RGD-Sm\u03b11/Sm\u03b21 contact frequency (freq>{args.freq_threshold}, n={n_ok})"
                save_contact_map(hot_freq, csv_path=freq_csv, png_path=freq_png, title=title, binary=False, vmin=0.0, vmax=1.0)

                write_hotspot_pairs(
                    hot_freq,
                    os.path.join(args.contact_map_dir, f"contact_hotspot_pairs_{tag}.csv"),
                    min_value=float(args.freq_threshold),
                )
                save_hotspot_summaries(hot_freq, args.contact_map_dir, f"frequency_{tag}")

                if args.contact_frequency in ("by_ligand", "grid"):
                    freq_maps_for_grid: Dict[str, pd.DataFrame] = {}
                    n_by_lig: Dict[str, int] = {}

                    for lig, maps in bin_maps_by_lig.items():
                        if len(maps) < int(args.min_models_per_ligand):
                            continue
                        lig_freq, lig_n = aggregate_contact_frequency(maps)
                        lig_hot = compute_frequency_hotspots(lig_freq, threshold=float(args.freq_threshold))
                        if lig_hot.empty:
                            continue

                        lig_safe = re.sub(r"[^A-Za-z0-9_-]+", "_", lig)
                        lig_csv = os.path.join(args.contact_map_dir, f"contact_frequency_{lig_safe}_{tag}.csv")
                        lig_png = os.path.join(args.contact_map_dir, f"contact_frequency_{lig_safe}_{tag}.png") if args.plots else None
                        lig_title = f"RGD-Sm\u03b11/Sm\u03b21 contact frequency ({lig}, freq>{args.freq_threshold}, n={lig_n})"
                        save_contact_map(lig_hot, csv_path=lig_csv, png_path=lig_png, title=lig_title, binary=False, vmin=0.0, vmax=1.0)

                        freq_maps_for_grid[lig] = lig_hot
                        n_by_lig[lig] = lig_n

                    if args.contact_frequency == "grid" and args.plots and freq_maps_for_grid:
                        grid_png = os.path.join(args.contact_map_dir, f"contact_frequency_grid_{tag}.png")
                        plot_contact_frequency_grid(
                            freq_maps_for_grid,
                            n_by_lig,
                            grid_png,
                            threshold=float(args.freq_threshold),
                            vmax=1.0,
                            cmap="viridis",
                            ncols=int(args.ligand_ncols),
                        )

                print(f"Contact maps/frequency outputs saved to: {args.contact_map_dir}/")

    # Top-N CSV
    if args.top_n and args.top_n > 0:
        top_df = df.head(args.top_n)
        top_path = f"RGD_top{args.top_n}.csv"
        top_df.to_csv(top_path, index=False)
        print(f"Top {args.top_n} saved to: {top_path}")

    # Plots
    if args.plots:
        make_plots(df, outdir="plots", top_n=args.top_n if args.top_n > 0 else min(20, len(df)))
        print("Plots saved to: plots/")

    # Report
    if args.report:
        write_report(top, path="RGD_top_report.txt")
        print("Top model report saved to: RGD_top_report.txt")


if __name__ == "__main__":
    main()
