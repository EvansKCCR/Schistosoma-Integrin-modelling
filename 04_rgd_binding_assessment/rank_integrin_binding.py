#!/usr/bin/env python3
"""
Rank RGD–integrin binder models by composite biophysical metrics.

Enhancements:
- FreeSASA BSA via PDBWriter + temporary files (robust across environments)
- Portable chain selection (chainID with segid fallback)
- Robust Mg2+ auto-detection (with diagnostics) and optional chain hint
- Configurable score weights and MIDAS windows
- Optional Top-N CSV, plots (PNG), and text report
- Optional faster contact counting using capped_distance

"""

import os
import glob
import argparse
import logging
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
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# Optional FreeSASA
try:
    import freesasa  # noqa: F401
    HAVE_FREESASA = True
except Exception:
    HAVE_FREESASA = False


# ------------------------------------------------------------
# DEFAULT CONSTANTS (can be overridden via CLI or weights)
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


# ------------------------------------------------------------
# Utility / parsing helpers
# ------------------------------------------------------------
def parse_weights(s: Optional[str]) -> None:
    """
    Parse weight string like 'bsa=0.01,salt=2,hbond=1.5,contacts=0.1,midas=3'
    and update globals.
    """
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
            logging.warning(f"Ignoring unknown weight key: '{key}'")
            continue
        try:
            fval = float(val)
        except ValueError:
            logging.warning(f"Ignoring non-numeric weight value: '{p}'")
            continue
        globals()[mapping[key]] = fval


def select_chains(u: mda.Universe, chains: List[str]) -> AtomGroup:
    """
    Return atoms belonging to any of the given chains.
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
# Metric computations
# ------------------------------------------------------------
def compute_contacts(rgd: AtomGroup,
                     receptor: AtomGroup,
                     cutoff: float = CONTACT_CUTOFF,
                     heavy_only: bool = True,
                     fast: bool = False) -> int:
    """
    Count atom–atom contacts within cutoff (Å).
    If fast=True, uses capped_distance for better scaling on large systems.
    """
    if heavy_only:
        rgd = rgd.select_atoms("not name H*")
        receptor = receptor.select_atoms("not name H*")
    if len(rgd) == 0 or len(receptor) == 0:
        return 0

    if fast:
        # Returns pairs (i,j) within cutoff; length is the contact count
        pairs = capped_distance(rgd.positions, receptor.positions, cutoff, return_distances=False)
        return int(len(pairs))
    else:
        d = distance_array(rgd.positions, receptor.positions)
        return int(np.count_nonzero(d < cutoff))


def compute_bsa_proxy(contact_count: int) -> float:
    """Very rough BSA proxy (contacts × 6 Å²)."""
    return float(contact_count) * 6.0


def compute_bsa_freesasa(rgd: AtomGroup,
                         receptor: AtomGroup) -> Optional[float]:
    """
    Compute buried surface area using FreeSASA.

    BSA = SASA(RGD) + SASA(Receptor) - SASA(Complex)

    Uses MDAnalysis PDBWriter to temporary files (robust across environments).
    Returns None on any failure.
    """
    if not HAVE_FREESASA or len(rgd) == 0 or len(receptor) == 0:
        return None

    import tempfile

    complex_pdb = None
    rgd_pdb = None
    rec_pdb = None
    try:
        # Use delete=False for Windows/WSL semantics
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f_complex, \
             tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f_rgd, \
             tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f_rec:
            complex_pdb, rgd_pdb, rec_pdb = f_complex.name, f_rgd.name, f_rec.name

        # Write PDB files
        with PDBWriter(complex_pdb) as w:
            w.write(rgd | receptor)
        with PDBWriter(rgd_pdb) as w:
            w.write(rgd)
        with PDBWriter(rec_pdb) as w:
            w.write(receptor)

        # Run FreeSASA
        sasa_complex = freesasa.calc(freesasa.Structure(complex_pdb)).totalArea()
        sasa_rgd = freesasa.calc(freesasa.Structure(rgd_pdb)).totalArea()
        sasa_rec = freesasa.calc(freesasa.Structure(rec_pdb)).totalArea()

        bsa = (sasa_rgd + sasa_rec) - sasa_complex
        return max(float(bsa), 0.0)

    except Exception as e:
        logging.debug(f"FreeSASA failed: {e}")
        return None
    finally:
        # Clean up temp files
        for fn in (complex_pdb, rgd_pdb, rec_pdb):
            if fn and os.path.exists(fn):
                try:
                    os.unlink(fn)
                except Exception:
                    pass


def compute_salt_bridges(rgd: AtomGroup,
                         receptor: AtomGroup,
                         cutoff: float = SALT_CUTOFF) -> int:
    """
    Count salt-bridge close contacts based on sidechain charged atoms.
    Positive: ARG (NH1,NH2,NE), LYS (NZ)
    Negative: ASP (OD1,OD2), GLU (OE1,OE2)
    """
    positive = rgd.select_atoms("resname ARG LYS and name NH1 NH2 NE NZ")
    negative = receptor.select_atoms("resname ASP GLU and name OD1 OD2 OE1 OE2")
    if len(positive) == 0 or len(negative) == 0:
        return 0
    d = distance_array(positive.positions, negative.positions)
    return int(np.count_nonzero(d < cutoff))


def compute_hbonds(rgd: AtomGroup,
                   receptor: AtomGroup,
                   cutoff: float = HBOND_CUTOFF,
                   bidirectional: bool = False) -> int:
    """
    Simple heavy-atom donor/acceptor count (distance-only).
    Default: RGD donors -> receptor acceptors.
    With bidirectional=True: also counts receptor donors -> RGD acceptors.
    """
    donors_rgd = rgd.select_atoms("name N NE NH1 NH2 ND1 ND2 NZ")
    accept_receptor = receptor.select_atoms("name O OD1 OD2 OE1 OE2 OG OG1")
    primary = 0
    if len(donors_rgd) > 0 and len(accept_receptor) > 0:
        d = distance_array(donors_rgd.positions, accept_receptor.positions)
        primary = int(np.count_nonzero(d < cutoff))

    if not bidirectional:
        return primary

    donors_rec = receptor.select_atoms("name N NE NH1 NH2 ND1 ND2 NZ")
    accept_rgd = rgd.select_atoms("name O OD1 OD2 OE1 OE2 OG OG1")
    secondary = 0
    if len(donors_rec) > 0 and len(accept_rgd) > 0:
        d = distance_array(donors_rec.positions, accept_rgd.positions)
        secondary = int(np.count_nonzero(d < cutoff))

    return primary + secondary


def _pick_mg_atom(u: mda.Universe,
                  rgd_chain: str,
                  receptor_chains: List[str],
                  mg_chain: Optional[str]) -> Optional[Atom]:
    """
    Robust Mg selection:
    1) All atoms with (element Mg or resname/name MG)
    2) If mg_chain provided, try to narrow by chainID/segid
    3) If multiple remain, pick Mg closest to the centroid of (RGD Asp O + receptor pocket O)
    """
    mg_candidates = u.select_atoms("(element Mg) or (resname MG) or (name MG)")
    if mg_chain:
        narrowed = u.select_atoms(
            f"(chainID {mg_chain} or segid {mg_chain}) and ((element Mg) or (resname MG) or (name MG))"
        )
        if len(narrowed) > 0:
            mg_candidates = narrowed

    if len(mg_candidates) == 0:
        return None
    if len(mg_candidates) == 1:
        return mg_candidates[0]

    rgd_asp = u.select_atoms(
        f"(chainID {rgd_chain} or segid {rgd_chain}) and resname ASP and name OD1 OD2"
    )
    receptor_ox = u.select_atoms(
        f"(chainID {' '.join(receptor_chains)} or segid {' '.join(receptor_chains)}) "
        f"and name OD1 OD2 OE1 OE2 OG OG1"
    )
    env = (rgd_asp | receptor_ox)
    if len(env) == 0:
        return mg_candidates[0]

    centroid = env.positions.mean(axis=0)
    idx = np.argmin(np.linalg.norm(mg_candidates.positions - centroid, axis=1))
    return mg_candidates[idx]


def compute_midas_score(u: mda.Universe,
                        rgd_chain: str,
                        receptor_chains: List[str],
                        mg_chain: Optional[str],
                        mg_opt: Tuple[float, float],
                        mg_acc: Tuple[float, float]) -> Tuple[float, Dict[str, float]]:
    """
    Heuristic MIDAS integrity score (+ diagnostics):
      +5 if best Asp(O)–Mg in [opt_low,opt_high] Å
      +2 if in [acc_low,acc_high] Å
      -3 if > 3.0 Å
      +1 per receptor oxygen within acc_high Å of Mg
      -3 if coordinating oxygens < 4; -1 if > 6
      -variance(coordinating distances) penalty if > 1 oxygen
    Returns (score, diagnostics dict)
    """
    rgd_asp = u.select_atoms(
        f"(chainID {rgd_chain} or segid {rgd_chain}) and resname ASP and name OD1 OD2"
    )
    receptor_ox = u.select_atoms(
        f"(chainID {' '.join(receptor_chains)} or segid {' '.join(receptor_chains)}) "
        f"and name OD1 OD2 OE1 OE2 OG OG1"
    )
    mg_atom = _pick_mg_atom(u, rgd_chain, receptor_chains, mg_chain)

    diag = {
        "AspMg_min_dist": np.nan,
        "Mg_coord_count": 0,
        "Mg_coord_var": np.nan,
    }

    if mg_atom is None:
        return 0.0, diag

    score = 0.0

    # Asp–Mg
    if len(rgd_asp) > 0:
        dists = np.linalg.norm(rgd_asp.positions - mg_atom.position, axis=1)
        best = float(np.min(dists))
        diag["AspMg_min_dist"] = best
        if mg_opt[0] <= best <= mg_opt[1]:
            score += 5.0
        elif mg_acc[0] <= best <= mg_acc[1]:
            score += 2.0
        elif best > 3.0:
            score -= 3.0

    # Receptor–Mg coordination
    if len(receptor_ox) > 0:
        dists = np.linalg.norm(receptor_ox.positions - mg_atom.position, axis=1)
        coordinating = dists[dists < mg_acc[1]]
        coord_count = int(coordinating.size)
        diag["Mg_coord_count"] = coord_count
        if coord_count > 1:
            diag["Mg_coord_var"] = float(np.var(coordinating))
        score += float(coord_count)
        if coord_count < 4:
            score -= 3.0
        if coord_count > 6:
            score -= 1.0
        if coord_count > 1:
            score -= float(np.var(coordinating))

    return float(score), diag


# ------------------------------------------------------------
# Composite score
# ------------------------------------------------------------
def compute_biological_score(contacts: int,
                             bsa: float,
                             salt: int,
                             hbonds: int,
                             midas_score: float) -> float:
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
def analyze_model(pdb: str,
                  rgd_chains: List[str],
                  receptor_chains: List[str],
                  mg_chain: Optional[str],
                  use_freesasa: bool,
                  heavy_only: bool,
                  hbonds_bidirectional: bool,
                  fast_contacts: bool,
                  mg_opt: Tuple[float, float],
                  mg_acc: Tuple[float, float]) -> Optional[Dict[str, float]]:
    try:
        u = mda.Universe(pdb)
        rgd = select_chains(u, rgd_chains)
        receptor = select_chains(u, receptor_chains)

        if len(rgd) == 0 or len(receptor) == 0:
            logging.warning(f"Skipping {pdb}: empty RGD or receptor selection.")
            return None

        contacts = compute_contacts(
            rgd, receptor, cutoff=CONTACT_CUTOFF, heavy_only=heavy_only, fast=fast_contacts
        )

        bsa = None
        if use_freesasa and HAVE_FREESASA:
            bsa = compute_bsa_freesasa(rgd, receptor)
        if bsa is None:
            bsa = compute_bsa_proxy(contacts)

        salt = compute_salt_bridges(rgd, receptor, cutoff=SALT_CUTOFF)
        hbonds = compute_hbonds(rgd, receptor, cutoff=HBOND_CUTOFF, bidirectional=hbonds_bidirectional)

        midas_score, diag = compute_midas_score(u, rgd_chains[0], receptor_chains, mg_chain, mg_opt, mg_acc)

        bio_score = compute_biological_score(contacts, bsa, salt, hbonds, midas_score)

        out = {
            "model": os.path.basename(pdb),
            "contacts": contacts,
            "BSA": float(bsa),
            "salt_bridges": salt,
            "H_bonds": hbonds,
            "MIDAS_score": float(midas_score),
            "Biological_score": float(bio_score),
            # Diagnostics
            "AspMg_min_dist": diag["AspMg_min_dist"],
            "Mg_coord_count": diag["Mg_coord_count"],
            "Mg_coord_var": diag["Mg_coord_var"],
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
        f.write(f"  Biological_score : {top_row['Biological_score']:.6f}\n")
        f.write(f"  BSA (Å^2)        : {top_row['BSA']:.3f}\n")
        f.write(f"  Contacts         : {top_row['contacts']}\n")
        f.write(f"  H_bonds          : {top_row['H_bonds']}\n")
        f.write(f"  Salt_bridges     : {top_row['salt_bridges']}\n")
        f.write(f"  MIDAS_score      : {top_row['MIDAS_score']:.6f}\n")
        f.write("\nMIDAS diagnostics:\n")
        f.write(f"  Asp–Mg min dist (Å): {top_row['AspMg_min_dist']}\n")
        f.write(f"  Mg coord count     : {top_row['Mg_coord_count']}\n")
        f.write(f"  Mg coord variance  : {top_row['Mg_coord_var']}\n")
        f.write("\nNotes:\n")
        f.write("  - BSA computed via FreeSASA if available and --use_freesasa set.\n")
        f.write("  - H-bonds counted by heavy-atom distance heuristic.\n")
        f.write("  - MIDAS score uses tunable distance windows and coordination heuristics.\n")


def make_plots(df: pd.DataFrame, outdir: str, top_n: int) -> None:
    os.makedirs(outdir, exist_ok=True)
    # Scatter: score vs BSA
    plt.figure(figsize=(6, 4))
    plt.scatter(df["BSA"], df["Biological_score"], c="tab:blue", alpha=0.7)
    plt.xlabel("BSA (Å²)")
    plt.ylabel("Biological score")
    plt.title("Biological score vs BSA")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "score_vs_BSA.png"), dpi=200)
    plt.close()

    # Scatter: score vs H-bonds
    plt.figure(figsize=(6, 4))
    plt.scatter(df["H_bonds"], df["Biological_score"], c="tab:green", alpha=0.7)
    plt.xlabel("H-bonds (count)")
    plt.ylabel("Biological score")
    plt.title("Biological score vs H-bonds")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "score_vs_Hbonds.png"), dpi=200)
    plt.close()

    # Scatter: score vs contacts
    plt.figure(figsize=(6, 4))
    plt.scatter(df["contacts"], df["Biological_score"], c="tab:orange", alpha=0.7)
    plt.xlabel("Contacts (count)")
    plt.ylabel("Biological score")
    plt.title("Biological score vs contacts")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "score_vs_contacts.png"), dpi=200)
    plt.close()

    # Bar: Top-N models by score
    head = df.head(top_n) if top_n > 0 else df
    plt.figure(figsize=(max(6, 0.45 * len(head)), 4))
    plt.bar(head["model"], head["Biological_score"], color="tab:purple")
    plt.ylabel("Biological score")
    plt.title(f"Top {len(head)} models")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "topN_scores_bar.png"), dpi=200)
    plt.close()


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Rank RGD–integrin binders by composite biophysical criteria.")
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
    p.add_argument("--log", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--weights", default=None,
                   help="Override weights, e.g. 'bsa=0.01,salt=2,hbond=1.5,contacts=0.1,midas=3'")
    p.add_argument("--top_n", type=int, default=0, help="Export top-N rows to 'RGD_topN.csv' (0=all).")
    p.add_argument("--plots", action="store_true", help="Save quick plots (PNG) to 'plots/' directory.")
    p.add_argument("--report", action="store_true", help="Write a text report for the top model.")
    # Geometry overrides
    p.add_argument("--contact_cutoff", type=float, default=CONTACT_CUTOFF)
    p.add_argument("--salt_cutoff", type=float, default=SALT_CUTOFF)
    p.add_argument("--hbond_cutoff", type=float, default=HBOND_CUTOFF)
    p.add_argument("--midas_opt", type=float, nargs=2, default=list(MG_OPTIMAL), metavar=("LOW", "HIGH"),
                   help="Optimal Asp–Mg distance window (Å)")
    p.add_argument("--midas_acc", type=float, nargs=2, default=list(MG_ACCEPTABLE), metavar=("LOW", "HIGH"),
                   help="Acceptable Asp–Mg distance window (Å)")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log), format="%(levelname)s: %(message)s")

    # Override globals from CLI
    global CONTACT_CUTOFF, SALT_CUTOFF, HBOND_CUTOFF, MG_OPTIMAL, MG_ACCEPTABLE
    CONTACT_CUTOFF = args.contact_cutoff
    SALT_CUTOFF = args.salt_cutoff
    HBOND_CUTOFF = args.hbond_cutoff
    MG_OPTIMAL = tuple(args.midas_opt)
    MG_ACCEPTABLE = tuple(args.midas_acc)

    parse_weights(args.weights)

    if not os.path.isdir(args.input_dir):
        logging.error(f"Input dir not found: {args.input_dir}")
        return

    pdb_files = sorted(glob.glob(os.path.join(args.input_dir, "*.pdb")))
    if not pdb_files:
        logging.error(f"No PDB files found in: {args.input_dir}")
        return

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

    df = pd.DataFrame(results).sort_values(by="Biological_score", ascending=False)
    df.to_csv(args.output, index=False)

    top = df.iloc[0]
    print("\nMost biologically favored RGD binder:")
    print(top.to_string())
    print("\nFull ranking saved to:", args.output)

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