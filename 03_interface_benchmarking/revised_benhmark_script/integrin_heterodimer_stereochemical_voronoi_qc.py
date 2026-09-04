#!/usr/bin/env python3
"""
Integrin heterodimer stereochemical, backbone and Voronoi/VoroMQA quality pipeline
=================================================================================

Purpose
-------
This script is a heterodimer-aware modification of the original single-protein
validation workflow. It keeps the original MolProbity, CaBLAM and VoroMQA logic,
but adds explicit handling of alpha/beta chain pairs, interface residues and
heterodimer-level summaries.

Main analyses
-------------
1. Stereochemical quality
   - MolProbity score
   - clashscore
   - Ramachandran favored/outlier percentages
   - rotamer outlier percentage
   - C-beta deviations
   - bond/angle RMSD when reported by MolProbity
   - computed globally and, if requested, separately for alpha and beta chains

2. Backbone quality
   - CaBLAM disfavored/outlier/severe percentages when CaBLAM/phenix.cablam is installed
   - internal fallback backbone checks from the PDB itself:
       * residues missing backbone atoms N/CA/C/O
       * C-alpha chain-break gaps
       * peptide C-N bond outliers
       * omega/torsion non-planar outliers
       * cis-peptide count
   - computed globally and, where meaningful, chain-aware

3. Voronoi/VoroMQA packing quality
   - global VoroMQA dark/light score
   - per-residue VoroMQA local-score table parsed from VoroMQA score PDB B-factors
   - chain-level VoroMQA summaries
   - alpha-beta interface VoroMQA summaries
   - optional user-defined region summaries

4. Heterodimer/interface context
   - alpha-beta interface residues using NeighborSearch
   - residue-pair contacts
   - atom contacts
   - interchain clash proxy
   - interface confidence from B-factor/pLDDT column is reported, but not used in the heterodimer-context score

Required external tools for full output
---------------------------------------
- voronota-js-voromqa
- molprobity or phenix.molprobity
- cablam or phenix.cablam

The script remains usable even when these executables are absent: unavailable
metrics are reported as missing/error columns, while internal backbone/interface
checks are still computed.

Example
-------
python integrin_heterodimer_stereochemical_voronoi_qc.py \
    --input-dir heterodimer_models \
    --output-dir heterodimer_validation_results \
    --alpha-chain A \
    --beta-chain B \
    --region-residues metal_site_residues.txt \
    --nproc 4

Region residue file format
--------------------------
One residue per line, using chain-aware residue identifiers:
A:123
B:210
B:211A   # insertion code if present

Author/context
--------------
Adapted for integrin heterodimer structural validation.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import tempfile
import traceback
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
    from Bio.PDB import PDBParser, PDBIO, Select, NeighborSearch
    from Bio.PDB.Polypeptide import is_aa
    from Bio.PDB.vectors import calc_dihedral
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Biopython is required. Install with: pip install biopython\n"
        f"Import error: {exc}"
    )


# =============================================================================
# Constants
# =============================================================================

STANDARD_BACKBONE = ("N", "CA", "C", "O")
CONTACT_CUTOFF_DEFAULT = 5.0
CLASH_CUTOFF_DEFAULT = 2.0
CA_GAP_CUTOFF_DEFAULT = 4.5
PEPTIDE_BOND_MIN = 1.20
PEPTIDE_BOND_MAX = 1.50
OMEGA_PLANAR_TOL = 30.0


# =============================================================================
# Generic utilities
# =============================================================================

def tool_exists(name: str) -> bool:
    return shutil.which(name) is not None


def run_cmd(cmd: List[str], timeout: Optional[int] = None) -> str:
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout[:2000]}\n"
            f"STDERR:\n{proc.stderr[:4000]}"
        )
    return proc.stdout


def safe_float(value) -> float:
    try:
        if value is None:
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def zscore(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() < 2:
        return pd.Series(np.nan, index=s.index)
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    z = (s - mu) / sd
    return z if higher_is_better else -z


def mean_available(row: pd.Series, cols: List[str]) -> float:
    vals = [safe_float(row.get(c)) for c in cols]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan


# =============================================================================
# PDB cleaning and selections
# =============================================================================

def clean_pdb_file(
    inp: Path,
    out: Path,
    keep_hetatm: bool = False,
    keep_ter: bool = True,
    first_model_only: bool = True,
    altloc_policy: str = "A",
    remove_hydrogens: bool = True,
):
    """Clean PDB while retaining chain IDs and residue numbering."""
    altloc_policy = altloc_policy.lower()

    def keep_altloc(line: str) -> bool:
        alt = line[16:17].strip()
        if altloc_policy == "all":
            return True
        if altloc_policy == "blank":
            return alt == ""
        return alt in ("", "A")

    in_model = not first_model_only

    with inp.open("r", errors="ignore") as fin, out.open("w") as fout:
        for line in fin:
            rec = line[:6].strip()

            if first_model_only:
                if rec == "MODEL":
                    in_model = True
                    fout.write(line)
                    continue
                if rec == "ENDMDL":
                    break
                if rec not in ("ATOM", "HETATM", "TER", "END") and not in_model:
                    continue

            if rec in ("ATOM", "HETATM"):
                if rec == "HETATM" and not keep_hetatm:
                    continue
                if remove_hydrogens and (line[76:78].strip() == "H" or line[12:16].strip().startswith("H")):
                    continue
                if not keep_altloc(line):
                    continue
                fout.write(line)
            elif rec == "TER" and keep_ter:
                fout.write(line)
            elif rec in ("MODEL", "END"):
                fout.write(line)

        if keep_ter:
            fout.write("END\n")


def get_structure(path: Path):
    return PDBParser(QUIET=True).get_structure(path.stem, str(path))


def residue_uid(residue) -> str:
    chain = residue.get_parent().id
    rid = residue.get_id()
    resseq = rid[1]
    icode = rid[2].strip()
    return f"{chain}:{resseq}{icode}" if icode else f"{chain}:{resseq}"


def residue_label(residue) -> str:
    return f"{residue_uid(residue)}:{residue.get_resname()}"


def standard_residues(chain) -> List:
    return [r for r in chain if is_aa(r, standard=True) and r.get_id()[0] == " "]


class ChainSelect(Select):
    def __init__(self, chain_id: str):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return 1 if chain.id == self.chain_id else 0

    def accept_residue(self, residue):
        return 1 if residue.get_id()[0] == " " and is_aa(residue, standard=True) else 0

    def accept_atom(self, atom):
        return 0 if atom.element == "H" else 1


def write_chain_pdb(structure, chain_id: str, out_path: Path):
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_path), ChainSelect(chain_id))


# =============================================================================
# MolProbity parsing
# =============================================================================

def find_molprobity_cmd() -> Optional[str]:
    for cmd in ("molprobity", "phenix.molprobity"):
        if shutil.which(cmd):
            return cmd
    return None


def parse_molprobity(text: str) -> Dict[str, Optional[float]]:
    def extract(pattern: str, cast=float):
        m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            try:
                return cast(m.group(1))
            except Exception:
                return None
        return None

    return {
        "molprobity_score": extract(r"MolProbity\s+score\s*[:=]?\s*([\d\.]+)"),
        "clashscore": extract(r"Clashscore\s*[:=]?\s*([\d\.]+)"),
        "rama_favored_%": extract(r"(?:Ramachandran\s+)?Favored\s*[:=]\s*([\d\.]+)\s*%"),
        "rama_outliers_%": extract(r"(?:Ramachandran\s+)?Outliers\s*[:=]\s*([\d\.]+)\s*%"),
        "rotamer_outliers_%": extract(r"Rotamer.*?Outliers\s*[:=]\s*([\d\.]+)\s*%"),
        "cbeta_deviations": extract(r"C[- ]?beta\s+deviations\s*[:=]?\s*(\d+)", int),
        "rms_bonds": extract(r"RMS\(bonds\)\s*[:=]?\s*([\d\.]+)"),
        "rms_angles": extract(r"RMS\(angles\)\s*[:=]?\s*([\d\.]+)"),
        "cis_proline_%": extract(r"Cis-proline\s*[:=]\s*([\d\.]+)\s*%"),
        "cis_general_%": extract(r"Cis-general\s*[:=]\s*([\d\.]+)\s*%"),
    }


def empty_molprobity() -> Dict[str, Optional[float]]:
    return parse_molprobity("")


def run_molprobity(pdb: Path) -> Tuple[Dict[str, Optional[float]], Optional[str]]:
    cmd = find_molprobity_cmd()
    if not cmd:
        return empty_molprobity(), "MolProbity executable not found"
    try:
        return parse_molprobity(run_cmd([cmd, str(pdb)])), None
    except Exception as exc:
        return empty_molprobity(), str(exc)


# =============================================================================
# CaBLAM parsing
# =============================================================================

def find_cablam_cmd() -> Optional[str]:
    for cmd in ("cablam", "phenix.cablam"):
        if shutil.which(cmd):
            return cmd
    return None


def parse_cablam(text: str) -> Dict[str, Optional[float]]:
    data = {
        "cablam_disfavored_%": None,
        "cablam_outliers_%": None,
        "cablam_severe_%": None,
        "cablam_helix_%": None,
        "cablam_beta_%": None,
    }
    for line in text.splitlines():
        l = line.lower()
        m = re.search(r"\(([\d\.]+)%\)", line)
        if not m:
            m = re.search(r"([\d\.]+)\s*%", line)
        if not m:
            continue
        val = float(m.group(1))
        if "disfavored" in l:
            data["cablam_disfavored_%"] = val
        elif "outlier" in l and "severe" not in l:
            data["cablam_outliers_%"] = val
        elif "severe" in l:
            data["cablam_severe_%"] = val
        elif "helix-like" in l:
            data["cablam_helix_%"] = val
        elif "beta-like" in l:
            data["cablam_beta_%"] = val
    return data


def empty_cablam() -> Dict[str, Optional[float]]:
    return parse_cablam("")


def run_cablam(pdb: Path) -> Tuple[Dict[str, Optional[float]], Optional[str]]:
    cmd = find_cablam_cmd()
    if not cmd:
        return empty_cablam(), "CaBLAM executable not found"
    try:
        return parse_cablam(run_cmd([cmd, str(pdb)])), None
    except Exception as exc:
        return empty_cablam(), str(exc)


# =============================================================================
# Internal backbone quality checks
# =============================================================================

def atom_distance(a, b) -> float:
    return float(np.linalg.norm(a.coord - b.coord))


def residue_sort_key(res):
    rid = res.get_id()
    return (rid[1], rid[2])


def calc_omega_degrees(res_i, res_j) -> Optional[float]:
    required_i = ("CA", "C")
    required_j = ("N", "CA")
    if not all(a in res_i for a in required_i) or not all(a in res_j for a in required_j):
        return None
    try:
        omega = calc_dihedral(
            res_i["CA"].get_vector(),
            res_i["C"].get_vector(),
            res_j["N"].get_vector(),
            res_j["CA"].get_vector(),
        )
        return float(math.degrees(omega))
    except Exception:
        return None


def omega_class(omega: Optional[float]) -> str:
    if omega is None or np.isnan(omega):
        return "missing"
    a = abs(omega)
    if a <= OMEGA_PLANAR_TOL:
        return "cis"
    if abs(a - 180.0) <= OMEGA_PLANAR_TOL:
        return "trans"
    return "nonplanar_outlier"


def backbone_quality_for_structure(structure, chain_ids: Optional[Iterable[str]] = None) -> Tuple[Dict[str, float], List[Dict]]:
    """Return backbone quality summary and detailed residue/pair issues."""
    model = structure[0]
    if chain_ids is None:
        chain_ids = [c.id for c in model]
    chain_ids = [c for c in chain_ids if c in model]

    details: List[Dict] = []
    total_res = 0
    missing_backbone_res = 0
    ca_gap_count = 0
    peptide_bond_outlier_count = 0
    omega_nonplanar_count = 0
    omega_cis_count = 0
    peptide_pairs = 0

    for cid in chain_ids:
        chain = model[cid]
        residues = sorted(standard_residues(chain), key=residue_sort_key)
        total_res += len(residues)

        for res in residues:
            missing = [a for a in STANDARD_BACKBONE if a not in res]
            if missing:
                missing_backbone_res += 1
                details.append({
                    "chain": cid,
                    "residue_id": residue_uid(res),
                    "resname": res.get_resname(),
                    "issue_type": "missing_backbone_atoms",
                    "value": ";".join(missing),
                })

        for r1, r2 in zip(residues[:-1], residues[1:]):
            # only compare sequential residue numbers, but record gaps separately
            if "CA" in r1 and "CA" in r2:
                ca_dist = atom_distance(r1["CA"], r2["CA"])
                if ca_dist > CA_GAP_CUTOFF_DEFAULT:
                    ca_gap_count += 1
                    details.append({
                        "chain": cid,
                        "residue_id": f"{residue_uid(r1)}--{residue_uid(r2)}",
                        "resname": f"{r1.get_resname()}--{r2.get_resname()}",
                        "issue_type": "ca_gap_gt_cutoff",
                        "value": ca_dist,
                    })

            if "C" in r1 and "N" in r2:
                peptide_pairs += 1
                cn = atom_distance(r1["C"], r2["N"])
                if cn < PEPTIDE_BOND_MIN or cn > PEPTIDE_BOND_MAX:
                    peptide_bond_outlier_count += 1
                    details.append({
                        "chain": cid,
                        "residue_id": f"{residue_uid(r1)}--{residue_uid(r2)}",
                        "resname": f"{r1.get_resname()}--{r2.get_resname()}",
                        "issue_type": "peptide_cn_bond_outlier",
                        "value": cn,
                    })

            omega = calc_omega_degrees(r1, r2)
            oc = omega_class(omega)
            if oc == "cis":
                omega_cis_count += 1
                details.append({
                    "chain": cid,
                    "residue_id": f"{residue_uid(r1)}--{residue_uid(r2)}",
                    "resname": f"{r1.get_resname()}--{r2.get_resname()}",
                    "issue_type": "cis_peptide_geometry_proxy",
                    "value": omega,
                })
            elif oc == "nonplanar_outlier":
                omega_nonplanar_count += 1
                details.append({
                    "chain": cid,
                    "residue_id": f"{residue_uid(r1)}--{residue_uid(r2)}",
                    "resname": f"{r1.get_resname()}--{r2.get_resname()}",
                    "issue_type": "omega_nonplanar_outlier",
                    "value": omega,
                })

    denom_res = max(total_res, 1)
    denom_pairs = max(peptide_pairs, 1)
    summary = {
        "backbone_residues_checked": total_res,
        "missing_backbone_residue_count": missing_backbone_res,
        "missing_backbone_residue_%": 100.0 * missing_backbone_res / denom_res,
        "ca_gap_count": ca_gap_count,
        "ca_gap_per_100_res": 100.0 * ca_gap_count / denom_res,
        "peptide_pairs_checked": peptide_pairs,
        "peptide_cn_bond_outlier_count": peptide_bond_outlier_count,
        "peptide_cn_bond_outlier_%": 100.0 * peptide_bond_outlier_count / denom_pairs,
        "omega_nonplanar_outlier_count": omega_nonplanar_count,
        "omega_nonplanar_outlier_%": 100.0 * omega_nonplanar_count / denom_pairs,
        "omega_cis_count": omega_cis_count,
        "omega_cis_%": 100.0 * omega_cis_count / denom_pairs,
    }
    return summary, details


# =============================================================================
# Interface detection
# =============================================================================

def non_h_atoms(residue):
    return [a for a in residue.get_atoms() if getattr(a, "element", "") != "H"]


def detect_interface(structure, alpha_chain: str, beta_chain: str, contact_cutoff: float, clash_cutoff: float):
    model = structure[0]
    if alpha_chain not in model or beta_chain not in model:
        return {
            "interface_summary": {
                "alpha_chain_found": alpha_chain in model,
                "beta_chain_found": beta_chain in model,
                "alpha_interface_residue_count": 0,
                "beta_interface_residue_count": 0,
                "total_interface_residue_count": 0,
                "residue_pair_contacts": 0,
                "atom_pair_contacts": 0,
                "interchain_clash_proxy_count": 0,
            },
            "interface_residues": [],
            "interface_contacts": [],
        }

    atoms_a = [a for a in model[alpha_chain].get_atoms() if getattr(a, "element", "") != "H"]
    atoms_b = [a for a in model[beta_chain].get_atoms() if getattr(a, "element", "") != "H"]

    ns_b = NeighborSearch(atoms_b)

    alpha_res: Set[str] = set()
    beta_res: Set[str] = set()
    residue_pairs: Set[Tuple[str, str]] = set()
    contact_rows = []
    atom_pair_contacts = 0
    clash_proxy_count = 0

    for atom_a in atoms_a:
        for atom_b in ns_b.search(atom_a.coord, contact_cutoff, level="A"):
            res_a = atom_a.get_parent()
            res_b = atom_b.get_parent()
            if not is_aa(res_a, standard=True) or not is_aa(res_b, standard=True):
                continue
            uid_a = residue_uid(res_a)
            uid_b = residue_uid(res_b)
            dist = atom_distance(atom_a, atom_b)
            alpha_res.add(uid_a)
            beta_res.add(uid_b)
            residue_pairs.add((uid_a, uid_b))
            atom_pair_contacts += 1
            if dist <= clash_cutoff:
                clash_proxy_count += 1
            contact_rows.append({
                "alpha_residue": uid_a,
                "alpha_resname": res_a.get_resname(),
                "alpha_atom": atom_a.get_name(),
                "beta_residue": uid_b,
                "beta_resname": res_b.get_resname(),
                "beta_atom": atom_b.get_name(),
                "distance_A": dist,
                "is_clash_proxy": dist <= clash_cutoff,
            })

    interface_residue_rows = []
    for uid in sorted(alpha_res, key=lambda x: (x.split(":")[0], int(re.sub(r"\D", "", x.split(":")[1]) or 0), x)):
        interface_residue_rows.append({"chain_role": "alpha", "chain": alpha_chain, "residue_id": uid})
    for uid in sorted(beta_res, key=lambda x: (x.split(":")[0], int(re.sub(r"\D", "", x.split(":")[1]) or 0), x)):
        interface_residue_rows.append({"chain_role": "beta", "chain": beta_chain, "residue_id": uid})

    summary = {
        "alpha_chain_found": True,
        "beta_chain_found": True,
        "alpha_interface_residue_count": len(alpha_res),
        "beta_interface_residue_count": len(beta_res),
        "total_interface_residue_count": len(alpha_res) + len(beta_res),
        "residue_pair_contacts": len(residue_pairs),
        "atom_pair_contacts": atom_pair_contacts,
        "interchain_clash_proxy_count": clash_proxy_count,
        "contact_density_residue_pairs_per_interface_residue": len(residue_pairs) / max(len(alpha_res) + len(beta_res), 1),
        "atom_contact_density_per_interface_residue": atom_pair_contacts / max(len(alpha_res) + len(beta_res), 1),
        "interchain_clash_proxy_per_100_atom_contacts": 100.0 * clash_proxy_count / max(atom_pair_contacts, 1),
    }

    return {
        "interface_summary": summary,
        "interface_residues": interface_residue_rows,
        "interface_contacts": contact_rows,
    }


# =============================================================================
# B-factor/pLDDT summaries
# =============================================================================

def bfactor_summary(structure, residue_ids: Optional[Set[str]] = None, chain_ids: Optional[Set[str]] = None) -> Dict[str, float]:
    vals = []
    for res in structure.get_residues():
        if not is_aa(res, standard=True) or res.get_id()[0] != " ":
            continue
        if chain_ids is not None and res.get_parent().id not in chain_ids:
            continue
        if residue_ids is not None and residue_uid(res) not in residue_ids:
            continue
        for atom in res.get_atoms():
            if getattr(atom, "element", "") != "H":
                vals.append(float(atom.get_bfactor()))
    if not vals:
        return {"mean_bfactor_or_plddt": np.nan, "median_bfactor_or_plddt": np.nan, "min_bfactor_or_plddt": np.nan}
    arr = np.asarray(vals, dtype=float)
    return {
        "mean_bfactor_or_plddt": float(np.mean(arr)),
        "median_bfactor_or_plddt": float(np.median(arr)),
        "min_bfactor_or_plddt": float(np.min(arr)),
    }


# =============================================================================
# VoroMQA / Voronoi quality
# =============================================================================

# Notes on local-score handling
# -----------------------------
# The previous version incorrectly used --output-table-file from
# voronota-js-voromqa as if it were a per-residue table. In Voronota-JS this
# option writes a global-score table. Local residue scores are instead written
# into PDB B-factor fields using --output-dark-scores and --output-light-scores.
# This section therefore extracts residue-level VoroMQA scores from those local
# score PDBs and converts them to a clean chain-aware CSV.


def find_voromqa_cmd() -> Optional[str]:
    """Return the preferred available VoroMQA command."""
    for cmd in ("voronota-js-voromqa", "voronota-voromqa"):
        if shutil.which(cmd):
            return cmd
    return None


def run_voromqa_global(pdb: Path) -> Tuple[Dict[str, Optional[float]], Optional[str]]:
    cmd = find_voromqa_cmd()
    if not cmd:
        return {
            "voromqa_dark_score": None,
            "voromqa_light_score": None,
            "voromqa_global_score": None,
            "voromqa_command": None,
        }, "No VoroMQA command found: expected voronota-js-voromqa or voronota-voromqa"

    try:
        out = run_cmd([cmd, "--input", str(pdb)])

        # The JS wrapper normally prints a whitespace table containing dark and
        # light global scores. The classic wrapper prints one global score.
        nums = [float(x) for x in re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![A-Za-z])", out)]

        dark = None
        light = None
        global_score = None

        if cmd == "voronota-js-voromqa":
            # Existing outputs from this pipeline show dark then light scores.
            dark = nums[0] if len(nums) > 0 else None
            light = nums[1] if len(nums) > 1 else None
            global_score = dark if dark is not None else light
        else:
            # Classic voronota-voromqa reports a single global VoroMQA score.
            global_score = nums[0] if nums else None
            light = global_score

        return {
            "voromqa_dark_score": dark,
            "voromqa_light_score": light,
            "voromqa_global_score": global_score,
            "voromqa_command": cmd,
        }, None
    except Exception as exc:
        return {
            "voromqa_dark_score": None,
            "voromqa_light_score": None,
            "voromqa_global_score": None,
            "voromqa_command": cmd,
        }, str(exc)


def _residue_id_from_parts(chain: str, resseq: str, icode: str = "") -> str:
    chain = str(chain).strip()
    resseq = str(resseq).strip()
    icode = str(icode).strip()
    if icode in ("", ".", "?"):
        return f"{chain}:{resseq}"
    return f"{chain}:{resseq}{icode}"


def parse_voromqa_score_pdb(score_pdb: Path, score_name: str) -> pd.DataFrame:
    """Parse local VoroMQA score PDB where residue scores are stored as B-factors.

    VoroMQA local score PDBs may carry the same residue-level score on every atom
    of a residue. We aggregate by residue and keep mean/median/min/max so that
    the parser is stable even if atom-level values vary.
    """
    rows: List[Dict] = []
    if not score_pdb.exists() or score_pdb.stat().st_size == 0:
        return pd.DataFrame(columns=[
            "normalized_chain", "resnum", "icode", "normalized_residue_id",
            "normalized_resname", f"{score_name}_mean", f"{score_name}_median",
            f"{score_name}_min", f"{score_name}_max", f"{score_name}_atom_count",
        ])

    for line in score_pdb.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        try:
            chain = line[21].strip() or "_"
            resseq = line[22:26].strip()
            icode = line[26].strip()
            resname = line[17:20].strip()
            atom = line[12:16].strip()
            bfac_txt = line[60:66].strip()
            score = float(bfac_txt)
        except Exception:
            continue
        if not resseq:
            continue
        rows.append({
            "normalized_chain": chain,
            "resnum": resseq,
            "icode": icode,
            "normalized_residue_id": _residue_id_from_parts(chain, resseq, icode),
            "normalized_resname": resname,
            "atom": atom,
            "score": score,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "normalized_chain", "resnum", "icode", "normalized_residue_id",
            "normalized_resname", f"{score_name}_mean", f"{score_name}_median",
            f"{score_name}_min", f"{score_name}_max", f"{score_name}_atom_count",
        ])

    raw = pd.DataFrame(rows)
    group_cols = ["normalized_chain", "resnum", "icode", "normalized_residue_id", "normalized_resname"]
    agg = (
        raw.groupby(group_cols, dropna=False)["score"]
        .agg(["mean", "median", "min", "max", "count"])
        .reset_index()
        .rename(columns={
            "mean": f"{score_name}_mean",
            "median": f"{score_name}_median",
            "min": f"{score_name}_min",
            "max": f"{score_name}_max",
            "count": f"{score_name}_atom_count",
        })
    )
    return agg


def _merge_local_score_tables(tables: List[pd.DataFrame]) -> pd.DataFrame:
    tables = [t for t in tables if t is not None and not t.empty]
    if not tables:
        return pd.DataFrame()
    key_cols = ["normalized_chain", "resnum", "icode", "normalized_residue_id", "normalized_resname"]
    out = tables[0]
    for t in tables[1:]:
        out = pd.merge(out, t, on=key_cols, how="outer")
    # Stable numeric sorting by chain/residue number/insertion code.
    def _safe_int(x):
        m = re.search(r"-?\d+", str(x))
        return int(m.group(0)) if m else 0
    out["_sort_resnum"] = out["resnum"].map(_safe_int)
    out.sort_values(["normalized_chain", "_sort_resnum", "icode", "normalized_residue_id"], inplace=True)
    out.drop(columns=["_sort_resnum"], inplace=True)
    return out


def _file_has_pdb_atoms(path: Path) -> bool:
    """Return True only when a file contains ATOM/HETATM records."""
    try:
        if not path.exists() or path.stat().st_size == 0:
            return False
        with path.open("r", errors="ignore") as handle:
            for line in handle:
                if line.startswith(("ATOM  ", "HETATM")):
                    return True
    except Exception:
        return False
    return False


def _file_head(path: Path, n: int = 5) -> str:
    try:
        return "\n".join(path.read_text(errors="ignore").splitlines()[:n])
    except Exception:
        return ""


def _unlink_if_exists(path: Path):
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def run_voromqa_per_residue(pdb: Path, out_table: Path) -> Optional[str]:
    """Generate a strict chain-aware per-residue VoroMQA CSV.

    This function intentionally refuses global VoroMQA score tables. A valid
    local-score result must be a PDB file containing ATOM/HETATM records with
    local VoroMQA scores in the B-factor field. This prevents the common error
    where the global table from ``--output-table-file`` is misread as a
    per-residue table.
    """
    cmd = find_voromqa_cmd()
    if not cmd:
        return "No VoroMQA command found: expected voronota-js-voromqa or voronota-voromqa"

    out_table.parent.mkdir(parents=True, exist_ok=True)
    score_pdb_dir = out_table.parent / "local_score_pdb"
    score_pdb_dir.mkdir(exist_ok=True)
    stem = out_table.stem

    # Never allow stale per-residue output from an older run to survive.
    _unlink_if_exists(out_table)

    try:
        if cmd == "voronota-js-voromqa":
            dark_pdb = score_pdb_dir / f"{stem}_dark_scores.pdb"
            light_pdb = score_pdb_dir / f"{stem}_light_scores.pdb"
            table_out = score_pdb_dir / f"{stem}_global_scores.tsv"
            for f in (dark_pdb, light_pdb, table_out):
                _unlink_if_exists(f)

            # Official Voronota-JS options: --output-dark-scores and
            # --output-light-scores write PDB files with local scores as
            # B-factors; --output-table-file is retained only for global-score
            # diagnostics and is never parsed as per-residue data.
            run_cmd([
                cmd,
                "--input", str(pdb),
                "--output-table-file", str(table_out),
                "--output-dark-scores", str(dark_pdb),
                "--output-light-scores", str(light_pdb),
                "--order-by-residue-id",
            ])

            # Some installations expose only-global VoroMQA more reliably for
            # writing score PDBs. Fall back to it if the first command did not
            # produce true PDB files.
            if not (_file_has_pdb_atoms(dark_pdb) or _file_has_pdb_atoms(light_pdb)):
                og_cmd = shutil.which("voronota-js-only-global-voromqa")
                if og_cmd:
                    for f in (dark_pdb, light_pdb):
                        _unlink_if_exists(f)
                    run_cmd([
                        "voronota-js-only-global-voromqa",
                        "--input", str(pdb),
                        "--output-dark-pdb", str(dark_pdb),
                        "--output-light-pdb", str(light_pdb),
                    ])

            if not (_file_has_pdb_atoms(dark_pdb) or _file_has_pdb_atoms(light_pdb)):
                msg = (
                    "VoroMQA local-score PDB files were not produced. "
                    "The available output appears to be a global score table, not per-residue data. "
                    f"dark_head={_file_head(dark_pdb)!r}; light_head={_file_head(light_pdb)!r}; "
                    f"table_head={_file_head(table_out)!r}"
                )
                return msg

            tables = []
            if _file_has_pdb_atoms(dark_pdb):
                tables.append(parse_voromqa_score_pdb(dark_pdb, "voromqa_dark_score"))
            if _file_has_pdb_atoms(light_pdb):
                tables.append(parse_voromqa_score_pdb(light_pdb, "voromqa_light_score"))
            vdf = _merge_local_score_tables(tables)

        else:
            residue_pdb = score_pdb_dir / f"{stem}_residue_scores.pdb"
            _unlink_if_exists(residue_pdb)
            run_cmd([
                cmd,
                "--input", str(pdb),
                "--output-residue-scores-pdb", str(residue_pdb),
            ])
            if not _file_has_pdb_atoms(residue_pdb):
                return (
                    "Classic VoroMQA did not produce a residue-score PDB. "
                    f"output_head={_file_head(residue_pdb)!r}"
                )
            vdf = parse_voromqa_score_pdb(residue_pdb, "voromqa_residue_score")

        if vdf.empty:
            return "VoroMQA local-score PDB was created but no residue scores could be parsed"

        vdf["voromqa_local_score_source"] = cmd
        vdf.to_csv(out_table, index=False)
        return None

    except Exception as exc:
        _unlink_if_exists(out_table)
        return str(exc)


def validate_voromqa_local_table(
    df: pd.DataFrame,
    alpha_chain: Optional[str] = None,
    beta_chain: Optional[str] = None,
    min_residues: int = 20,
) -> Optional[str]:
    """Reject global VoroMQA tables masquerading as residue-level output."""
    if df is None or df.empty:
        return "VoroMQA residue table is empty"
    required = {"normalized_residue_id", "normalized_chain"}
    missing = required.difference(df.columns)
    if missing:
        return f"VoroMQA residue table lacks required columns: {sorted(missing)}"

    n_unique = df["normalized_residue_id"].astype(str).nunique(dropna=True)
    if n_unique < min_residues:
        return (
            f"Rejected VoroMQA table with only {n_unique} unique residue IDs; "
            "this is almost certainly the global VoroMQA score table, not local per-residue output."
        )

    chains = set(df["normalized_chain"].astype(str))
    wanted = {str(c) for c in (alpha_chain, beta_chain) if c is not None}
    if wanted and chains.isdisjoint(wanted):
        return (
            f"Rejected VoroMQA table because detected chains {sorted(chains)} do not include "
            f"the requested alpha/beta chains {sorted(wanted)}."
        )

    score_col = pick_residue_score_column(df)
    if not score_col:
        return "VoroMQA residue table has no usable numeric local-score column"
    return None


def first_existing_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def normalize_voromqa_residue_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize arbitrary VoroMQA local score tables to chain-aware residue IDs."""
    df = df.copy()

    # Tables written by this script already have normalized fields.
    if "normalized_residue_id" in df.columns and "normalized_chain" in df.columns:
        df["normalized_residue_id"] = df["normalized_residue_id"].astype(str)
        df["normalized_chain"] = df["normalized_chain"].astype(str)
        return df

    chain_col = first_existing_col(df, ["chain", "chain_id", "chainID", "asym_id", "ID_chainID"])
    resnum_col = first_existing_col(df, ["resnum", "rnum", "residue_number", "res_seq", "resseq", "seqnum", "ID_resSeq"])
    icode_col = first_existing_col(df, ["icode", "iCode", "insertion_code", "ID_iCode"])
    resid_col = first_existing_col(df, ["residue_id", "residue", "id", "residueId"])
    resname_col = first_existing_col(df, ["resname", "residue_name", "residue_type", "name", "ID_resName"])

    def norm_row(row):
        if resid_col is not None and pd.notna(row.get(resid_col)):
            raw = str(row.get(resid_col)).strip()
            m = re.search(r"([A-Za-z0-9_])[:_\-\s]+(-?\d+[A-Za-z]?)", raw)
            if m:
                return f"{m.group(1)}:{m.group(2)}"

        chain = str(row.get(chain_col)).strip() if chain_col is not None and pd.notna(row.get(chain_col)) else ""
        resn = None
        if resnum_col is not None and pd.notna(row.get(resnum_col)):
            m = re.search(r"(-?\d+)", str(row.get(resnum_col)))
            if m:
                resn = m.group(1)
        elif resid_col is not None and pd.notna(row.get(resid_col)):
            m = re.search(r"(-?\d+[A-Za-z]?)", str(row.get(resid_col)))
            if m:
                resn = m.group(1)

        icode = ""
        if icode_col is not None and pd.notna(row.get(icode_col)):
            icode = str(row.get(icode_col)).strip()
            if icode in (".", "?", "nan"):
                icode = ""

        if chain and resn:
            return _residue_id_from_parts(chain, resn, icode)
        if resn:
            return str(resn)
        return None

    df["normalized_residue_id"] = df.apply(norm_row, axis=1)
    if chain_col is not None:
        df["normalized_chain"] = df[chain_col].astype(str).replace({".": "", "nan": ""})
    else:
        df["normalized_chain"] = df["normalized_residue_id"].astype(str).str.split(":", n=1).str[0]
    if resname_col is not None:
        df["normalized_resname"] = df[resname_col].astype(str)
    return df


def read_voromqa_table(path: Path) -> pd.DataFrame:
    lines = path.read_text(errors="ignore").splitlines()
    if not lines:
        return pd.DataFrame()
    first = lines[0]
    if "," in first:
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, sep=r"\s+", engine="python")
    return normalize_voromqa_residue_ids(df)


def pick_residue_score_column(df: pd.DataFrame) -> Optional[str]:
    """Select the most interpretable local VoroMQA residue score column."""
    preferred = [
        "voromqa_light_score_mean",  # classic/light method, preferred for continuity
        "voromqa_dark_score_mean",   # newer dark method
        "voromqa_residue_score_mean",
        "light_score",
        "dark_score",
        "residue_score",
    ]
    for c in preferred:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any():
            return c

    # Fall back only to numeric columns that look like scores. Avoid IDs/counts.
    bad_patterns = ("count", "resnum", "index", "serial", "atom_count")
    for c in df.columns:
        cl = c.lower()
        if "score" in cl and not any(b in cl for b in bad_patterns):
            if pd.to_numeric(df[c], errors="coerce").notna().any():
                return c
    return None


def summarize_numeric_distribution(vals: pd.Series, prefix: str) -> Dict[str, float]:
    x = pd.to_numeric(vals, errors="coerce").dropna()
    if x.empty:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_q10": np.nan,
            f"{prefix}_q90": np.nan,
            f"{prefix}_worst10_mean": np.nan,
        }
    q10 = x.quantile(0.10)
    q90 = x.quantile(0.90)
    return {
        f"{prefix}_n": int(x.shape[0]),
        f"{prefix}_mean": float(x.mean()),
        f"{prefix}_median": float(x.median()),
        f"{prefix}_q10": float(q10),
        f"{prefix}_q90": float(q90),
        f"{prefix}_worst10_mean": float(x[x <= q10].mean()),
    }


def load_residue_set(path: Optional[Path]) -> Set[str]:
    if path is None:
        return set()
    residues = set()
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        residues.add(line)
    return residues


def voromqa_summaries(
    df: pd.DataFrame,
    score_col: str,
    alpha_chain: str,
    beta_chain: str,
    interface_ids: Set[str],
    region_ids: Set[str],
) -> Dict[str, float]:
    out: Dict[str, float] = {"voromqa_residue_score_column": score_col}
    out.update(summarize_numeric_distribution(df[score_col], "voromqa_all_residue"))

    alpha_df = df[df["normalized_chain"].astype(str) == str(alpha_chain)]
    beta_df = df[df["normalized_chain"].astype(str) == str(beta_chain)]
    out.update(summarize_numeric_distribution(alpha_df[score_col] if not alpha_df.empty else pd.Series(dtype=float), "voromqa_alpha_chain"))
    out.update(summarize_numeric_distribution(beta_df[score_col] if not beta_df.empty else pd.Series(dtype=float), "voromqa_beta_chain"))

    if interface_ids:
        iface_df = df[df["normalized_residue_id"].astype(str).isin(interface_ids)]
    else:
        iface_df = pd.DataFrame(columns=df.columns)
    out.update(summarize_numeric_distribution(iface_df[score_col] if not iface_df.empty else pd.Series(dtype=float), "voromqa_interface"))

    if region_ids:
        region_df = df[df["normalized_residue_id"].astype(str).isin(region_ids)]
        out.update(summarize_numeric_distribution(region_df[score_col] if not region_df.empty else pd.Series(dtype=float), "voromqa_region"))
    else:
        out.update({
            "voromqa_region_n": 0,
            "voromqa_region_mean": np.nan,
            "voromqa_region_median": np.nan,
            "voromqa_region_q10": np.nan,
            "voromqa_region_q90": np.nan,
            "voromqa_region_worst10_mean": np.nan,
        })

    return out

# =============================================================================
# Prefix helper
# =============================================================================

def prefix_dict(d: Dict, prefix: str) -> Dict:
    return {f"{prefix}_{k}": v for k, v in d.items()}


# =============================================================================
# Worker
# =============================================================================

@dataclass
class Config:
    output_dir: Path
    cleaned_dir: Path
    chain_dir: Path
    per_res_dir: Path
    alpha_chain: str
    beta_chain: str
    contact_cutoff: float
    clash_cutoff: float
    keep_hetatm: bool
    keep_metals: bool
    altloc_policy: str
    remove_hydrogens: bool
    region_residues: Optional[Path]
    run_chain_tools: bool


def process_pdb(pdb: Path, cfg: Config) -> Dict:
    row: Dict = {"model": pdb.name, "status": "ok"}
    perres_rows: List[Dict] = []
    interface_residue_rows: List[Dict] = []
    interface_contact_rows: List[Dict] = []
    backbone_issue_rows: List[Dict] = []
    chain_quality_rows: List[Dict] = []

    try:
        cleaned = cfg.cleaned_dir / pdb.name
        clean_pdb_file(
            pdb,
            cleaned,
            keep_hetatm=cfg.keep_hetatm or cfg.keep_metals,
            keep_ter=True,
            first_model_only=True,
            altloc_policy=cfg.altloc_policy,
            remove_hydrogens=cfg.remove_hydrogens,
        )

        structure = get_structure(cleaned)
        model = structure[0]
        chains_present = [c.id for c in model]
        row["chains_present"] = ";".join(chains_present)
        row["alpha_chain"] = cfg.alpha_chain
        row["beta_chain"] = cfg.beta_chain
        row["alpha_chain_found"] = cfg.alpha_chain in chains_present
        row["beta_chain_found"] = cfg.beta_chain in chains_present

        # ------------------------------------------------------------------
        # Interface context
        # ------------------------------------------------------------------
        iface = detect_interface(
            structure,
            cfg.alpha_chain,
            cfg.beta_chain,
            cfg.contact_cutoff,
            cfg.clash_cutoff,
        )
        row.update(iface["interface_summary"])

        for r in iface["interface_residues"]:
            rr = {"model": pdb.name, **r}
            interface_residue_rows.append(rr)
        for r in iface["interface_contacts"]:
            rr = {"model": pdb.name, **r}
            interface_contact_rows.append(rr)

        interface_ids = {r["residue_id"] for r in iface["interface_residues"]}
        region_ids = load_residue_set(cfg.region_residues)

        # B-factor/pLDDT context
        row.update(prefix_dict(bfactor_summary(structure), "complex"))
        row.update(prefix_dict(bfactor_summary(structure, residue_ids=interface_ids), "interface"))
        row.update(prefix_dict(bfactor_summary(structure, chain_ids={cfg.alpha_chain}), "alpha_chain"))
        row.update(prefix_dict(bfactor_summary(structure, chain_ids={cfg.beta_chain}), "beta_chain"))

        # ------------------------------------------------------------------
        # Stereochemical quality: MolProbity, complex-level
        # ------------------------------------------------------------------
        mp, mp_err = run_molprobity(cleaned)
        row.update(prefix_dict(mp, "complex"))
        if mp_err:
            row["complex_molprobity_error"] = mp_err

        # ------------------------------------------------------------------
        # Backbone quality: CaBLAM plus internal fallback checks, complex-level
        # ------------------------------------------------------------------
        cb, cb_err = run_cablam(cleaned)
        row.update(prefix_dict(cb, "complex"))
        if cb_err:
            row["complex_cablam_error"] = cb_err

        bb_summary, bb_details = backbone_quality_for_structure(structure)
        row.update(prefix_dict(bb_summary, "complex"))
        for d in bb_details:
            backbone_issue_rows.append({"model": pdb.name, "scope": "complex", **d})

        # ------------------------------------------------------------------
        # Optional alpha/beta chain-level MolProbity/CaBLAM/backbone summaries
        # ------------------------------------------------------------------
        if cfg.run_chain_tools:
            for role, cid in [("alpha", cfg.alpha_chain), ("beta", cfg.beta_chain)]:
                if cid not in chains_present:
                    continue
                chain_pdb = cfg.chain_dir / f"{pdb.stem}_{role}_{cid}.pdb"
                write_chain_pdb(structure, cid, chain_pdb)

                chain_row = {"model": pdb.name, "chain_role": role, "chain": cid, "chain_pdb": chain_pdb.name}

                mp_c, mp_c_err = run_molprobity(chain_pdb)
                chain_row.update(mp_c)
                if mp_c_err:
                    chain_row["molprobity_error"] = mp_c_err

                cb_c, cb_c_err = run_cablam(chain_pdb)
                chain_row.update(cb_c)
                if cb_c_err:
                    chain_row["cablam_error"] = cb_c_err

                bb_c_summary, bb_c_details = backbone_quality_for_structure(structure, chain_ids=[cid])
                chain_row.update(bb_c_summary)
                chain_quality_rows.append(chain_row)
                row.update(prefix_dict(bb_c_summary, f"{role}_chain"))

                for d in bb_c_details:
                    backbone_issue_rows.append({"model": pdb.name, "scope": f"{role}_chain", **d})

        # ------------------------------------------------------------------
        # VoroMQA global and per-residue Voronoi packing quality
        # ------------------------------------------------------------------
        vglob, vglob_err = run_voromqa_global(cleaned)
        row.update(prefix_dict(vglob, "complex"))
        if vglob_err:
            row["complex_voromqa_error"] = vglob_err

        per_res_file = cfg.per_res_dir / f"{pdb.stem}_voromqa_per_residue.csv"
        per_err = run_voromqa_per_residue(cleaned, per_res_file)
        if per_err:
            row["per_residue_voromqa_error"] = per_err
        elif per_res_file.exists():
            vdf = read_voromqa_table(per_res_file)
            validation_error = validate_voromqa_local_table(vdf, cfg.alpha_chain, cfg.beta_chain)
            if validation_error:
                row["per_residue_voromqa_error"] = validation_error
            else:
                score_col = pick_residue_score_column(vdf)
                row.update(voromqa_summaries(vdf, score_col, cfg.alpha_chain, cfg.beta_chain, interface_ids, region_ids))
                for _, rr in vdf.iterrows():
                    record = rr.to_dict()
                    record["model"] = pdb.name
                    record["selected_voromqa_score_col"] = score_col
                    record["selected_voromqa_score"] = rr.get(score_col)
                    record["is_alpha_beta_interface"] = str(rr.get("normalized_residue_id")) in interface_ids
                    record["is_user_region"] = str(rr.get("normalized_residue_id")) in region_ids if region_ids else False
                    perres_rows.append(record)

        # ------------------------------------------------------------------
        # Store detailed rows in sidecar CSVs for this worker
        # ------------------------------------------------------------------
        sidecar_dir = cfg.output_dir / "_worker_sidecars"
        sidecar_dir.mkdir(exist_ok=True)
        stem = pdb.stem.replace(os.sep, "_")
        pd.DataFrame(perres_rows).to_csv(sidecar_dir / f"{stem}_perres.csv", index=False)
        pd.DataFrame(interface_residue_rows).to_csv(sidecar_dir / f"{stem}_interface_residues.csv", index=False)
        pd.DataFrame(interface_contact_rows).to_csv(sidecar_dir / f"{stem}_interface_contacts.csv", index=False)
        pd.DataFrame(backbone_issue_rows).to_csv(sidecar_dir / f"{stem}_backbone_issues.csv", index=False)
        pd.DataFrame(chain_quality_rows).to_csv(sidecar_dir / f"{stem}_chain_quality.csv", index=False)

    except Exception:
        row["status"] = "failed"
        row["traceback"] = traceback.format_exc(limit=20)

    return row


# =============================================================================
# Merge sidecars and compute ranking indices
# =============================================================================

def concat_sidecars(sidecar_dir: Path, suffix: str, out_path: Path):
    frames = []
    for f in sorted(sidecar_dir.glob(f"*_{suffix}.csv")):
        try:
            if f.stat().st_size > 0:
                df = pd.read_csv(f)
                if not df.empty:
                    frames.append(df)
        except Exception:
            pass
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)
    else:
        pd.DataFrame().to_csv(out_path, index=False)


def compute_quality_indices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Stereochemical: lower MolProbity/clash/outliers is better, higher favored is better.
    stereo_components = []
    stereo_specs = [
        ("complex_molprobity_score", False),
        ("complex_clashscore", False),
        ("complex_rama_favored_%", True),
        ("complex_rama_outliers_%", False),
        ("complex_rotamer_outliers_%", False),
        ("complex_cbeta_deviations", False),
    ]
    for col, hib in stereo_specs:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            zcol = f"{col}_z"
            df[zcol] = zscore(df[col], higher_is_better=hib)
            stereo_components.append(zcol)
    df["stereochemical_quality_index"] = df.apply(lambda r: mean_available(r, stereo_components), axis=1) if stereo_components else np.nan

    # Backbone: lower CaBLAM and internal outlier/gap percentages are better.
    backbone_components = []
    backbone_specs = [
        ("complex_cablam_disfavored_%", False),
        ("complex_cablam_outliers_%", False),
        ("complex_cablam_severe_%", False),
        ("complex_missing_backbone_residue_%", False),
        ("complex_ca_gap_per_100_res", False),
        ("complex_peptide_cn_bond_outlier_%", False),
        ("complex_omega_nonplanar_outlier_%", False),
    ]
    for col, hib in backbone_specs:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            zcol = f"{col}_z"
            df[zcol] = zscore(df[col], higher_is_better=hib)
            backbone_components.append(zcol)
    df["backbone_quality_index"] = df.apply(lambda r: mean_available(r, backbone_components), axis=1) if backbone_components else np.nan

    # Voronoi/VoroMQA: VoroMQA scores are interpreted as higher-is-better packing scores.
    voronoi_components = []
    voronoi_specs = [
        ("complex_voromqa_dark_score", True),
        ("complex_voromqa_light_score", True),
        ("voromqa_all_residue_mean", True),
        ("voromqa_interface_mean", True),
        ("voromqa_interface_worst10_mean", True),
    ]
    for col, hib in voronoi_specs:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            zcol = f"{col}_z"
            df[zcol] = zscore(df[col], higher_is_better=hib)
            voronoi_components.append(zcol)
    df["voronoi_quality_index"] = df.apply(lambda r: mean_available(r, voronoi_components), axis=1) if voronoi_components else np.nan

    # Heterodimer interface penalties/bonuses from internal context.
    interface_components = []
    interface_specs = [
        ("contact_density_residue_pairs_per_interface_residue", True),
        ("interchain_clash_proxy_per_100_atom_contacts", False),
    ]
    for col, hib in interface_specs:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            zcol = f"{col}_z"
            df[zcol] = zscore(df[col], higher_is_better=hib)
            interface_components.append(zcol)
    df["heterodimer_context_quality_index"] = df.apply(lambda r: mean_available(r, interface_components), axis=1) if interface_components else np.nan

    final_cols = [
        "stereochemical_quality_index",
        "backbone_quality_index",
        "voronoi_quality_index",
        "heterodimer_context_quality_index",
    ]
    df["heterodimer_validation_index"] = df.apply(lambda r: mean_available(r, final_cols), axis=1)
    if "heterodimer_validation_index" in df.columns:
        df.sort_values("heterodimer_validation_index", ascending=False, inplace=True, na_position="last")

    return df


def write_tool_status(out_dir: Path):
    rows = [
        {"tool": "molprobity", "available_command": find_molprobity_cmd()},
        {"tool": "cablam", "available_command": find_cablam_cmd()},
        {"tool": "VoroMQA", "available_command": find_voromqa_cmd()},
    ]
    pd.DataFrame(rows).to_csv(out_dir / "tool_status.csv", index=False)


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Stereochemical, backbone and Voronoi/VoroMQA quality analysis for integrin heterodimer complexes."
    )
    ap.add_argument("--input-dir", default="pdb_files", help="Folder containing heterodimer PDB files")
    ap.add_argument("--output-dir", default="heterodimer_validation_results", help="Output folder")
    ap.add_argument("--alpha-chain", default="A", help="Chain ID for alpha integrin chain")
    ap.add_argument("--beta-chain", default="B", help="Chain ID for beta integrin chain")
    ap.add_argument("--region-residues", default=None, help="Optional chain-aware residue list for motifs/regions, e.g. metal sites")
    ap.add_argument("--contact-cutoff", type=float, default=CONTACT_CUTOFF_DEFAULT, help="Alpha-beta atom contact cutoff in Å")
    ap.add_argument("--clash-cutoff", type=float, default=CLASH_CUTOFF_DEFAULT, help="Interchain clash proxy cutoff in Å")
    ap.add_argument("--keep-hetatm", action="store_true", help="Retain HETATM records")
    ap.add_argument("--keep-metals", action="store_true", help="Retain HETATM records for metal-aware validation runs")
    ap.add_argument("--keep-hydrogens", action="store_true", help="Do not remove hydrogen atoms during cleaning")
    ap.add_argument("--altloc-policy", default="A", choices=["A", "blank", "all"], help="Alternative-location handling")
    ap.add_argument("--skip-chain-tools", action="store_true", help="Skip separate chain-level MolProbity/CaBLAM runs")
    ap.add_argument("--nproc", type=int, default=max(1, cpu_count() - 1), help="Number of parallel workers")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    cleaned_dir = output_dir / "cleaned_pdb"
    chain_dir = output_dir / "chain_pdb"
    per_res_dir = output_dir / "per_residue_voromqa"
    sidecar_dir = output_dir / "_worker_sidecars"

    for d in (output_dir, cleaned_dir, chain_dir, per_res_dir, sidecar_dir):
        d.mkdir(parents=True, exist_ok=True)

    pdbs = sorted(input_dir.glob("*.pdb"))
    if not pdbs:
        raise SystemExit(f"No PDB files found in {input_dir}")

    cfg = Config(
        output_dir=output_dir,
        cleaned_dir=cleaned_dir,
        chain_dir=chain_dir,
        per_res_dir=per_res_dir,
        alpha_chain=args.alpha_chain,
        beta_chain=args.beta_chain,
        contact_cutoff=args.contact_cutoff,
        clash_cutoff=args.clash_cutoff,
        keep_hetatm=args.keep_hetatm,
        keep_metals=args.keep_metals,
        altloc_policy=args.altloc_policy,
        remove_hydrogens=not args.keep_hydrogens,
        region_residues=Path(args.region_residues) if args.region_residues else None,
        run_chain_tools=not args.skip_chain_tools,
    )

    write_tool_status(output_dir)

    if args.nproc <= 1:
        rows = [process_pdb(p, cfg) for p in pdbs]
    else:
        with Pool(args.nproc) as pool:
            rows = pool.starmap(process_pdb, [(p, cfg) for p in pdbs])

    summary = pd.DataFrame(rows)
    summary = compute_quality_indices(summary)
    summary.to_csv(output_dir / "heterodimer_validation_summary.csv", index=False)

    concat_sidecars(sidecar_dir, "perres", output_dir / "voromqa_per_residue_all_models.csv")
    concat_sidecars(sidecar_dir, "interface_residues", output_dir / "interface_residues_all_models.csv")
    concat_sidecars(sidecar_dir, "interface_contacts", output_dir / "interface_contacts_all_models.csv")
    concat_sidecars(sidecar_dir, "backbone_issues", output_dir / "backbone_issues_all_models.csv")
    concat_sidecars(sidecar_dir, "chain_quality", output_dir / "chain_quality_summary.csv")

    print("\nHeterodimer stereochemical/backbone/Voronoi validation complete.")
    print(f"Main summary: {output_dir / 'heterodimer_validation_summary.csv'}")
    print(f"Tool status:  {output_dir / 'tool_status.csv'}")
    print("Detailed tables:")
    print(f"  - {output_dir / 'chain_quality_summary.csv'}")
    print(f"  - {output_dir / 'voromqa_per_residue_all_models.csv'}")
    print(f"  - {output_dir / 'interface_residues_all_models.csv'}")
    print(f"  - {output_dir / 'interface_contacts_all_models.csv'}")
    print(f"  - {output_dir / 'backbone_issues_all_models.csv'}\n")


if __name__ == "__main__":
    main()
