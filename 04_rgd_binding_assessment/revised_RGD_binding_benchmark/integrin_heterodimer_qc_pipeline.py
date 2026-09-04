#!/usr/bin/env python3
"""
Integrin heterodimer fold-conservation, packing-quality, and interface-benchmarking pipeline.

Purpose
-------
This script is a heterodimer-focused repurposing of single-protein structural fold and
interface-quality workflows. It assumes that baseline chain-level model quality has already
been assessed elsewhere and focuses on:

1. Fold conservation
   - Alpha-chain and beta-chain fold benchmarking against templates/reference structures.
   - Optional domain-level benchmarking when a domain map is supplied.
   - Uses TMalign when available; otherwise falls back to exact-residue C-alpha RMSD and
     C-alpha contact-map overlap.

2. Heterodimer packing quality
   - Alpha-beta interface residues and residue-pair contacts.
   - Contact, H-bond proxy, salt-bridge proxy, hydrophobic-contact, and disulfide counts.
   - Interface buried surface area using Bio.PDB ShrakeRupley, with no FreeSASA dependency.
   - Mean interface B-factor/pLDDT if stored in the B-factor column.

3. Interface benchmarking
   - Optional native/reference-based AB RMSD, interface RMSD, ligand-chain RMSD proxy,
     native-contact recovery, non-native contact fraction, and contact Jaccard index.

Author context: Evans Asamoah Adu / integrin heterodimer structural assessment.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from Bio.PDB import NeighborSearch, PDBIO, PDBParser, Select, Superimposer
from Bio.PDB.SASA import ShrakeRupley

# -----------------------------
# Configuration constants
# -----------------------------

CONTACT_CUTOFF = 5.0
HBOND_DIST = 3.5
SALT_DIST = 4.0
HYDROPHOBIC_DIST = 4.5
DISULFIDE_MIN = 1.8
DISULFIDE_MAX = 2.4
CMAP_CUTOFF = 8.0
MIN_CA_PAIRS = 10

HYDROPHOBIC_3 = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TYR", "TRP", "PRO", "CYS"}
ACIDIC_ATOMS = {"ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"}}
BASIC_ATOMS = {"LYS": {"NZ"}, "ARG": {"NH1", "NH2", "NE"}, "HIS": {"ND1", "NE2"}}
STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
    "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "SEC", "PYL"
}

# -----------------------------
# Basic structure utilities
# -----------------------------


def parse_structure(path: Path):
    return PDBParser(QUIET=True).get_structure(path.stem, str(path))


def is_standard_residue(residue) -> bool:
    return residue.get_id()[0] == " " and residue.get_resname().strip() in STANDARD_AA


def residue_key(residue) -> Tuple[str, int, str]:
    hetflag, resseq, icode = residue.get_id()
    return residue.get_parent().id, int(resseq), icode.strip()


def residue_label(residue) -> str:
    chain, resseq, icode = residue_key(residue)
    suffix = icode if icode else ""
    return f"{chain}:{resseq}{suffix}:{residue.get_resname().strip()}"


def residue_pair_label(resA, resB) -> Tuple[str, str]:
    return residue_label(resA), residue_label(resB)


def heavy_atoms(chain) -> List:
    return [a for a in chain.get_atoms() if getattr(a, "element", "") != "H" and is_standard_residue(a.get_parent())]


def ca_atoms_for_chain(structure, chain_id: str, residue_numbers: Optional[Set[int]] = None) -> List:
    if chain_id not in structure[0]:
        return []
    atoms = []
    for res in structure[0][chain_id]:
        if not is_standard_residue(res) or "CA" not in res:
            continue
        if residue_numbers is not None and int(res.get_id()[1]) not in residue_numbers:
            continue
        atoms.append(res["CA"])
    return atoms


def ca_coords_for_chain(structure, chain_id: str, residue_numbers: Optional[Set[int]] = None) -> np.ndarray:
    atoms = ca_atoms_for_chain(structure, chain_id, residue_numbers)
    if not atoms:
        return np.empty((0, 3))
    return np.array([a.coord for a in atoms], dtype=float)


def model_chain_summary(structure, chain_id: str) -> Dict[str, float]:
    if chain_id not in structure[0]:
        return {"n_residues": 0, "n_ca": 0, "mean_bfactor_or_plddt": np.nan}
    residues = [r for r in structure[0][chain_id] if is_standard_residue(r)]
    ca = [r["CA"] for r in residues if "CA" in r]
    bvals = [a.get_bfactor() for r in residues for a in r.get_atoms()]
    return {
        "n_residues": len(residues),
        "n_ca": len(ca),
        "mean_bfactor_or_plddt": float(np.mean(bvals)) if bvals else np.nan,
    }

# -----------------------------
# Contact maps and fallback fold metrics
# -----------------------------


def contact_map(coords: np.ndarray, cutoff: float = CMAP_CUTOFF) -> np.ndarray:
    if len(coords) == 0:
        return np.zeros((0, 0), dtype=bool)
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    return dist <= cutoff


def contact_map_overlap(coords1: np.ndarray, coords2: np.ndarray) -> float:
    n = min(len(coords1), len(coords2))
    if n == 0:
        return np.nan
    c1 = contact_map(coords1[:n])
    c2 = contact_map(coords2[:n])
    denom = np.sum(c2)
    if denom == 0:
        return np.nan
    return float(np.sum(c1 & c2) / denom)


def exact_residue_ca_pairs(struct1, struct2, chain_id: str, residue_numbers: Optional[Set[int]] = None):
    if chain_id not in struct1[0] or chain_id not in struct2[0]:
        return [], []
    pred_atoms, ref_atoms = [], []
    ref_chain = struct2[0][chain_id]
    for res in struct1[0][chain_id]:
        if not is_standard_residue(res) or "CA" not in res:
            continue
        resseq = int(res.get_id()[1])
        if residue_numbers is not None and resseq not in residue_numbers:
            continue
        rid = res.get_id()
        if rid in ref_chain and "CA" in ref_chain[rid]:
            pred_atoms.append(res["CA"])
            ref_atoms.append(ref_chain[rid]["CA"])
    return pred_atoms, ref_atoms


def exact_residue_rmsd(struct1, struct2, chain_id: str, residue_numbers: Optional[Set[int]] = None) -> float:
    pred_atoms, ref_atoms = exact_residue_ca_pairs(struct1, struct2, chain_id, residue_numbers)
    if len(pred_atoms) < MIN_CA_PAIRS:
        return np.nan
    sup = Superimposer()
    sup.set_atoms(ref_atoms, pred_atoms)
    return float(sup.rms)


def q_score(n_align: int, l1: int, l2: int, rmsd: float, d0: float = 3.0) -> float:
    if n_align <= 0 or l1 <= 0 or l2 <= 0 or np.isnan(rmsd):
        return np.nan
    return float((n_align / math.sqrt(l1 * l2)) * (1 / (1 + (rmsd * rmsd / (d0 * d0)))))

# -----------------------------
# TMalign support
# -----------------------------


class ChainResidueSelect(Select):
    def __init__(self, chain_id: str, residue_numbers: Optional[Set[int]] = None):
        self.chain_id = chain_id
        self.residue_numbers = residue_numbers

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        if not is_standard_residue(residue):
            return False
        if self.residue_numbers is None:
            return True
        return int(residue.get_id()[1]) in self.residue_numbers

    def accept_atom(self, atom):
        return getattr(atom, "element", "") != "H"


def save_selection(structure, out_path: Path, chain_id: str, residue_numbers: Optional[Set[int]] = None) -> None:
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_path), ChainResidueSelect(chain_id, residue_numbers))


def find_tmalign() -> Optional[str]:
    for name in ("TMalign", "tmalign", "TM-align"):
        p = shutil.which(name)
        if p:
            return p
    return None


def parse_tmalign_output(text: str) -> Dict[str, float]:
    rmsd = np.nan
    alnlen = np.nan
    len1 = np.nan
    len2 = np.nan
    tm1 = np.nan
    tm2 = np.nan

    m = re.search(r"Aligned length=\s*(\d+),\s*RMSD=\s*([0-9.]+)", text)
    if m:
        alnlen = int(m.group(1))
        rmsd = float(m.group(2))
    else:
        m_aln = re.search(r"Aligned length=\s*(\d+)", text)
        m_rmsd = re.search(r"RMSD=\s*([0-9.]+)", text)
        if m_aln:
            alnlen = int(m_aln.group(1))
        if m_rmsd:
            rmsd = float(m_rmsd.group(1))

    m1 = re.search(r"Length of Chain_1:\s*(\d+)", text)
    m2 = re.search(r"Length of Chain_2:\s*(\d+)", text)
    if m1:
        len1 = int(m1.group(1))
    if m2:
        len2 = int(m2.group(1))

    tms = re.findall(r"TM-score=\s*([0-9.]+)", text)
    if len(tms) >= 1:
        tm1 = float(tms[0])
    if len(tms) >= 2:
        tm2 = float(tms[1])

    return {
        "tmalign_rmsd": rmsd,
        "tmalign_aligned_length": alnlen,
        "tmalign_len_model": len1,
        "tmalign_len_reference": len2,
        "tm_score_model_norm": tm1,
        "tm_score_reference_norm": tm2,
    }


def run_tmalign(model_pdb: Path, ref_pdb: Path) -> Dict[str, float]:
    exe = find_tmalign()
    if not exe:
        return {
            "tmalign_rmsd": np.nan,
            "tmalign_aligned_length": np.nan,
            "tmalign_len_model": np.nan,
            "tmalign_len_reference": np.nan,
            "tm_score_model_norm": np.nan,
            "tm_score_reference_norm": np.nan,
            "tmalign_error": "TMalign not found in PATH",
        }
    try:
        proc = subprocess.run([exe, str(model_pdb), str(ref_pdb)], text=True, capture_output=True, check=True)
        out = parse_tmalign_output(proc.stdout)
        out["tmalign_error"] = ""
        return out
    except Exception as e:
        return {
            "tmalign_rmsd": np.nan,
            "tmalign_aligned_length": np.nan,
            "tmalign_len_model": np.nan,
            "tmalign_len_reference": np.nan,
            "tm_score_model_norm": np.nan,
            "tm_score_reference_norm": np.nan,
            "tmalign_error": str(e),
        }

# -----------------------------
# Domain map
# -----------------------------


@dataclass
class DomainDef:
    domain: str
    chain: str
    start: int
    end: int

    @property
    def residue_numbers(self) -> Set[int]:
        return set(range(self.start, self.end + 1))


def read_domain_map(path: Optional[Path]) -> List[DomainDef]:
    if path is None:
        return []
    rows: List[DomainDef] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"domain", "chain", "start", "end"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Domain map must contain columns: domain,chain,start,end")
        for row in reader:
            rows.append(DomainDef(row["domain"], row["chain"], int(row["start"]), int(row["end"])))
    return rows

# -----------------------------
# Interface analysis
# -----------------------------


def detect_interface(structure, chainA: str, chainB: str, cutoff: float = CONTACT_CUTOFF):
    if chainA not in structure[0] or chainB not in structure[0]:
        raise ValueError(f"Missing chain(s): {chainA}, {chainB}")
    atomsA = heavy_atoms(structure[0][chainA])
    atomsB = heavy_atoms(structure[0][chainB])
    ns = NeighborSearch(atomsB)

    resA_set, resB_set = set(), set()
    residue_pairs: Set[Tuple[str, str]] = set()
    atom_contact_count = 0

    for atomA in atomsA:
        for atomB in ns.search(atomA.coord, cutoff):
            resA, resB = atomA.get_parent(), atomB.get_parent()
            resA_set.add(resA)
            resB_set.add(resB)
            residue_pairs.add(residue_pair_label(resA, resB))
            atom_contact_count += 1

    return resA_set, resB_set, residue_pairs, atom_contact_count


def count_hbonds_proxy(structure, residue_pairs: Set[Tuple[str, str]], chainA: str, chainB: str) -> int:
    count = 0
    model = structure[0]
    for labelA, labelB in residue_pairs:
        _, rnumA, _ = parse_residue_label(labelA)
        _, rnumB, _ = parse_residue_label(labelB)
        resA = get_residue_by_number(model[chainA], rnumA)
        resB = get_residue_by_number(model[chainB], rnumB)
        if resA is None or resB is None:
            continue
        for a in resA:
            if getattr(a, "element", "") not in ("N", "O"):
                continue
            for b in resB:
                if getattr(b, "element", "") not in ("N", "O"):
                    continue
                if a - b <= HBOND_DIST:
                    count += 1
    return count


def parse_residue_label(label: str) -> Tuple[str, int, str]:
    # Format chain:resseq[:icode]:resname; current residue_label emits chain:resseq:resname
    parts = label.split(":")
    chain = parts[0]
    resnum_part = parts[1]
    m = re.match(r"(\d+)", resnum_part)
    return chain, int(m.group(1)), parts[-1]


def get_residue_by_number(chain, resnum: int):
    for res in chain:
        if is_standard_residue(res) and int(res.get_id()[1]) == resnum:
            return res
    return None


def charged_atoms(residue) -> List:
    resname = residue.get_resname().strip()
    names = ACIDIC_ATOMS.get(resname, set()) | BASIC_ATOMS.get(resname, set())
    return [a for a in residue if a.get_name().strip() in names]


def count_salt_bridges_proxy(structure, chainA: str, chainB: str) -> int:
    model = structure[0]
    pairs = set()
    for resA in model[chainA]:
        if not is_standard_residue(resA):
            continue
        a_acidic = resA.get_resname().strip() in ACIDIC_ATOMS
        a_basic = resA.get_resname().strip() in BASIC_ATOMS
        if not (a_acidic or a_basic):
            continue
        for resB in model[chainB]:
            if not is_standard_residue(resB):
                continue
            b_acidic = resB.get_resname().strip() in ACIDIC_ATOMS
            b_basic = resB.get_resname().strip() in BASIC_ATOMS
            if not ((a_acidic and b_basic) or (a_basic and b_acidic)):
                continue
            for a in charged_atoms(resA):
                for b in charged_atoms(resB):
                    if a - b <= SALT_DIST:
                        pairs.add(residue_pair_label(resA, resB))
    return len(pairs)


def count_hydrophobic_contacts(structure, residue_pairs: Set[Tuple[str, str]], chainA: str, chainB: str) -> int:
    model = structure[0]
    pairs = set()
    for labelA, labelB in residue_pairs:
        _, rnumA, _ = parse_residue_label(labelA)
        _, rnumB, _ = parse_residue_label(labelB)
        resA = get_residue_by_number(model[chainA], rnumA)
        resB = get_residue_by_number(model[chainB], rnumB)
        if resA is None or resB is None:
            continue
        if resA.get_resname().strip() not in HYDROPHOBIC_3 or resB.get_resname().strip() not in HYDROPHOBIC_3:
            continue
        for a in resA:
            if getattr(a, "element", "") == "H":
                continue
            for b in resB:
                if getattr(b, "element", "") == "H":
                    continue
                if a - b <= HYDROPHOBIC_DIST:
                    pairs.add((labelA, labelB))
                    break
    return len(pairs)


def count_interchain_disulfides(structure, chainA: str, chainB: str) -> int:
    model = structure[0]
    count = 0
    cysA = [r for r in model[chainA] if is_standard_residue(r) and r.get_resname().strip() == "CYS" and "SG" in r]
    cysB = [r for r in model[chainB] if is_standard_residue(r) and r.get_resname().strip() == "CYS" and "SG" in r]
    for resA in cysA:
        for resB in cysB:
            d = resA["SG"] - resB["SG"]
            if DISULFIDE_MIN <= d <= DISULFIDE_MAX:
                count += 1
    return count


def mean_interface_bfactor(residues: Iterable) -> float:
    vals = [a.get_bfactor() for r in residues for a in r.get_atoms()]
    return float(np.mean(vals)) if vals else np.nan

# -----------------------------
# BSA by ShrakeRupley
# -----------------------------


class ChainOnlySelect(Select):
    def __init__(self, chain_id: str):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        return is_standard_residue(residue)

    def accept_atom(self, atom):
        return getattr(atom, "element", "") != "H"


class TwoChainSelect(Select):
    def __init__(self, chainA: str, chainB: str):
        self.chains = {chainA, chainB}

    def accept_chain(self, chain):
        return chain.id in self.chains

    def accept_residue(self, residue):
        return is_standard_residue(residue)

    def accept_atom(self, atom):
        return getattr(atom, "element", "") != "H"


def write_selected_pdb(structure, path: Path, select_obj: Select):
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(path), select_obj)


def total_sasa(path: Path) -> float:
    struct = parse_structure(path)
    sr = ShrakeRupley(n_points=100)
    sr.compute(struct, level="A")
    return float(sum(getattr(a, "sasa", 0.0) for a in struct.get_atoms()))


def compute_bsa_shrakerupley(structure, chainA: str, chainB: str) -> float:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        complex_p = td / "complex.pdb"
        chainA_p = td / "chainA.pdb"
        chainB_p = td / "chainB.pdb"
        write_selected_pdb(structure, complex_p, TwoChainSelect(chainA, chainB))
        write_selected_pdb(structure, chainA_p, ChainOnlySelect(chainA))
        write_selected_pdb(structure, chainB_p, ChainOnlySelect(chainB))
        return total_sasa(chainA_p) + total_sasa(chainB_p) - total_sasa(complex_p)

# -----------------------------
# Reference benchmarking
# -----------------------------


def joint_align_copy(pred_structure, ref_structure, chainA: str, chainB: str):
    pred = copy.deepcopy(pred_structure)
    pred_atoms, ref_atoms = [], []
    for chain_id in (chainA, chainB):
        if chain_id not in pred[0] or chain_id not in ref_structure[0]:
            continue
        ref_chain = ref_structure[0][chain_id]
        for res in pred[0][chain_id]:
            if not is_standard_residue(res) or "CA" not in res:
                continue
            rid = res.get_id()
            if rid in ref_chain and "CA" in ref_chain[rid]:
                pred_atoms.append(res["CA"])
                ref_atoms.append(ref_chain[rid]["CA"])
    if len(pred_atoms) < MIN_CA_PAIRS:
        return pred, np.nan, len(pred_atoms)
    sup = Superimposer()
    sup.set_atoms(ref_atoms, pred_atoms)
    sup.apply(pred.get_atoms())
    return pred, float(sup.rms), len(pred_atoms)


def rmsd_over_residue_labels(pred_aligned, ref_structure, labels: Set[str], chainA: str, chainB: str) -> float:
    pred_atoms, ref_atoms = [], []
    for label in labels:
        chain_id, resnum, _ = parse_residue_label(label)
        if chain_id not in (chainA, chainB):
            continue
        if chain_id not in pred_aligned[0] or chain_id not in ref_structure[0]:
            continue
        pred_res = get_residue_by_number(pred_aligned[0][chain_id], resnum)
        ref_res = get_residue_by_number(ref_structure[0][chain_id], resnum)
        if pred_res is None or ref_res is None or "CA" not in pred_res or "CA" not in ref_res:
            continue
        pred_atoms.append(pred_res["CA"])
        ref_atoms.append(ref_res["CA"])
    if len(pred_atoms) < 5:
        return np.nan
    diffs = np.array([p.coord - r.coord for p, r in zip(pred_atoms, ref_atoms)])
    return float(np.sqrt(np.mean(np.sum(diffs * diffs, axis=1))))


def chain_rmsd_after_partner_alignment(pred_structure, ref_structure, align_chain: str, mobile_chain: str) -> float:
    pred = copy.deepcopy(pred_structure)
    pred_atoms, ref_atoms = exact_residue_ca_pairs(pred, ref_structure, align_chain)
    if len(pred_atoms) < MIN_CA_PAIRS:
        return np.nan
    sup = Superimposer()
    sup.set_atoms(ref_atoms, pred_atoms)
    sup.apply(pred.get_atoms())
    mob_pred, mob_ref = exact_residue_ca_pairs(pred, ref_structure, mobile_chain)
    if len(mob_pred) < MIN_CA_PAIRS:
        return np.nan
    diffs = np.array([p.coord - r.coord for p, r in zip(mob_pred, mob_ref)])
    return float(np.sqrt(np.mean(np.sum(diffs * diffs, axis=1))))


def benchmark_against_reference(pred_structure, ref_structure, chainA: str, chainB: str) -> Dict[str, float]:
    pred_aligned, global_rmsd, n_align = joint_align_copy(pred_structure, ref_structure, chainA, chainB)

    pred_resA, pred_resB, pred_pairs, _ = detect_interface(pred_aligned, chainA, chainB)
    ref_resA, ref_resB, ref_pairs, _ = detect_interface(ref_structure, chainA, chainB)

    ref_interface_labels = {residue_label(r) for r in ref_resA | ref_resB}
    pred_interface_labels = {residue_label(r) for r in pred_resA | pred_resB}

    common_pairs = pred_pairs & ref_pairs
    fnat = len(common_pairs) / len(ref_pairs) if ref_pairs else np.nan
    non_native_fraction = (len(pred_pairs - ref_pairs) / len(pred_pairs)) if pred_pairs else np.nan
    jaccard = len(common_pairs) / len(pred_pairs | ref_pairs) if (pred_pairs | ref_pairs) else np.nan
    iface_rmsd_ref_iface = rmsd_over_residue_labels(pred_aligned, ref_structure, ref_interface_labels, chainA, chainB)
    iface_rmsd_common_iface = rmsd_over_residue_labels(pred_aligned, ref_structure, ref_interface_labels & pred_interface_labels, chainA, chainB)
    beta_ligand_rmsd_after_alpha = chain_rmsd_after_partner_alignment(pred_structure, ref_structure, chainA, chainB)
    alpha_ligand_rmsd_after_beta = chain_rmsd_after_partner_alignment(pred_structure, ref_structure, chainB, chainA)

    return {
        "Reference_Global_RMSD_AB": global_rmsd,
        "Reference_CA_pairs_aligned": n_align,
        "Reference_Interface_RMSD_native_iface": iface_rmsd_ref_iface,
        "Reference_Interface_RMSD_common_iface": iface_rmsd_common_iface,
        "Reference_Fnat_native_contacts": fnat,
        "Reference_non_native_contact_fraction": non_native_fraction,
        "Reference_contact_jaccard": jaccard,
        "Reference_common_contact_pairs": len(common_pairs),
        "Reference_native_contact_pairs": len(ref_pairs),
        "Reference_pred_contact_pairs": len(pred_pairs),
        "Reference_beta_RMSD_after_alpha_fit": beta_ligand_rmsd_after_alpha,
        "Reference_alpha_RMSD_after_beta_fit": alpha_ligand_rmsd_after_beta,
    }

# -----------------------------
# Fold conservation
# -----------------------------


def fold_comparison_rows(model_name: str, model_struct, templates: Dict[str, object], chain_ids: Sequence[str], domains: List[DomainDef], work_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    targets: List[Tuple[str, str, Optional[Set[int]]]] = []
    if domains:
        for d in domains:
            if d.chain in chain_ids:
                targets.append((d.domain, d.chain, d.residue_numbers))
    else:
        for c in chain_ids:
            targets.append((f"chain_{c}", c, None))

    for template_name, template_struct in templates.items():
        for target_name, chain_id, residue_numbers in targets:
            row = {
                "Model": model_name,
                "Template": template_name,
                "Fold_unit": target_name,
                "Chain": chain_id,
            }
            coords_model = ca_coords_for_chain(model_struct, chain_id, residue_numbers)
            coords_template = ca_coords_for_chain(template_struct, chain_id, residue_numbers)
            fallback_rmsd = exact_residue_rmsd(model_struct, template_struct, chain_id, residue_numbers)
            cmo = contact_map_overlap(coords_model, coords_template)
            row.update({
                "Fallback_exact_CA_RMSD": fallback_rmsd,
                "Fallback_CMO": cmo,
                "Fallback_Q_score": q_score(min(len(coords_model), len(coords_template)), len(coords_model), len(coords_template), fallback_rmsd),
                "Model_CA_count": len(coords_model),
                "Template_CA_count": len(coords_template),
            })

            with tempfile.TemporaryDirectory(dir=str(work_dir)) as td:
                td = Path(td)
                model_sel = td / "model_sel.pdb"
                ref_sel = td / "ref_sel.pdb"
                save_selection(model_struct, model_sel, chain_id, residue_numbers)
                save_selection(template_struct, ref_sel, chain_id, residue_numbers)
                row.update(run_tmalign(model_sel, ref_sel))

            rows.append(row)
    return rows

# -----------------------------
# Ranking and plotting
# -----------------------------


def zscore(series: pd.Series, invert: bool = False) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        z = pd.Series(0.0, index=s.index)
    else:
        z = (s - mu) / sd
    return -z if invert else z


def add_packing_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df
    score_specs = {
        "Interface_residues": False,
        "Residue_pair_contacts": False,
        "Contact_density": False,
        "Hbond_density": False,
        "Salt_bridge_density": False,
        "Hydrophobic_contact_density": False,
        "BSA_total_A2": False,
        "Interface_mean_bfactor_or_plddt": False,
    }
    for col, invert in score_specs.items():
        if col in df:
            df[f"{col}_z"] = zscore(df[col], invert=invert)

    zcols = [c for c in df.columns if c.endswith("_z")]
    if zcols:
        weights = {
            "Interface_residues_z": 0.10,
            "Residue_pair_contacts_z": 0.15,
            "Contact_density_z": 0.20,
            "Hbond_density_z": 0.15,
            "Salt_bridge_density_z": 0.10,
            "Hydrophobic_contact_density_z": 0.10,
            "BSA_total_A2_z": 0.15,
            "Interface_mean_bfactor_or_plddt_z": 0.05,
        }
        df["Packing_quality_index"] = 0.0
        used = 0.0
        for col, w in weights.items():
            if col in df:
                df["Packing_quality_index"] += df[col].fillna(0) * w
                used += w
        if used > 0:
            df["Packing_quality_index"] = df["Packing_quality_index"] / used
    else:
        df["Packing_quality_index"] = np.nan

    df.sort_values("Packing_quality_index", ascending=False, inplace=True)
    return df


def add_reference_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty or "Reference_Fnat_native_contacts" not in df:
        return df
    metrics = {
        "Reference_Fnat_native_contacts": False,
        "Reference_contact_jaccard": False,
        "Reference_non_native_contact_fraction": True,
        "Reference_Global_RMSD_AB": True,
        "Reference_Interface_RMSD_native_iface": True,
        "Reference_beta_RMSD_after_alpha_fit": True,
        "Reference_alpha_RMSD_after_beta_fit": True,
    }
    for col, invert in metrics.items():
        if col in df:
            df[f"{col}_z"] = zscore(df[col], invert=invert)
    zcols = [c for c in df.columns if c.startswith("Reference_") and c.endswith("_z")]
    if zcols:
        df["Reference_interface_benchmark_index"] = df[zcols].mean(axis=1)
    return df


def plot_bar(df: pd.DataFrame, metric: str, out_path: Path, title: str) -> None:
    if df.empty or metric not in df:
        return
    plot_df = df[["Model", metric]].dropna().sort_values(metric, ascending=False)
    if plot_df.empty:
        return
    plt.figure(figsize=(max(7, 0.45 * len(plot_df)), 5))
    plt.bar(plot_df["Model"], plot_df[metric])
    plt.ylabel(metric)
    plt.title(title)
    plt.xticks(rotation=70, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_metric_heatmap(df: pd.DataFrame, metrics: List[str], out_path: Path, title: str) -> None:
    use = [m for m in metrics if m in df.columns]
    if df.empty or not use:
        return
    plot_df = df.set_index("Model")[use].apply(pd.to_numeric, errors="coerce")
    if plot_df.empty:
        return
    norm = plot_df.copy()
    for c in norm.columns:
        s = norm[c]
        if s.max() != s.min():
            norm[c] = (s - s.min()) / (s.max() - s.min())
        else:
            norm[c] = 0.0
    plt.figure(figsize=(max(7, 0.5 * len(use)), max(4, 0.4 * len(norm))))
    plt.imshow(norm.values, aspect="auto")
    plt.yticks(range(len(norm.index)), norm.index)
    plt.xticks(range(len(use)), use, rotation=45, ha="right")
    plt.colorbar(label="min-max normalized value")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# -----------------------------
# Main model-level analysis
# -----------------------------


def analyze_packing(model_path: Path, structure, chainA: str, chainB: str) -> Dict[str, object]:
    resA, resB, residue_pairs, atom_contacts = detect_interface(structure, chainA, chainB)
    iface_size = len(resA) + len(resB)
    hbonds = count_hbonds_proxy(structure, residue_pairs, chainA, chainB)
    salts = count_salt_bridges_proxy(structure, chainA, chainB)
    hydrophobics = count_hydrophobic_contacts(structure, residue_pairs, chainA, chainB)
    disulfides = count_interchain_disulfides(structure, chainA, chainB)
    try:
        bsa = compute_bsa_shrakerupley(structure, chainA, chainB)
    except Exception:
        bsa = np.nan

    alpha_stats = model_chain_summary(structure, chainA)
    beta_stats = model_chain_summary(structure, chainB)

    return {
        "Model": model_path.name,
        "Alpha_chain": chainA,
        "Beta_chain": chainB,
        "Alpha_residues": alpha_stats["n_residues"],
        "Beta_residues": beta_stats["n_residues"],
        "Alpha_mean_bfactor_or_plddt": alpha_stats["mean_bfactor_or_plddt"],
        "Beta_mean_bfactor_or_plddt": beta_stats["mean_bfactor_or_plddt"],
        "Interface_residues_alpha": len(resA),
        "Interface_residues_beta": len(resB),
        "Interface_residues": iface_size,
        "Residue_pair_contacts": len(residue_pairs),
        "Atom_contact_count": atom_contacts,
        "Contact_density": len(residue_pairs) / max(iface_size, 1),
        "Hbonds_proxy": hbonds,
        "Hbond_density": hbonds / max(iface_size, 1),
        "Salt_bridges_proxy": salts,
        "Salt_bridge_density": salts / max(iface_size, 1),
        "Hydrophobic_contacts": hydrophobics,
        "Hydrophobic_contact_density": hydrophobics / max(iface_size, 1),
        "Interchain_disulfides": disulfides,
        "BSA_total_A2": bsa,
        "BSA_per_interface_residue_A2": bsa / max(iface_size, 1) if not pd.isna(bsa) else np.nan,
        "Interface_mean_bfactor_or_plddt": mean_interface_bfactor(resA | resB),
    }


def load_templates(templates_dir: Optional[Path]) -> Dict[str, object]:
    if templates_dir is None:
        return {}
    templates = {}
    for p in sorted(templates_dir.glob("*.pdb")):
        templates[p.name] = parse_structure(p)
    return templates


def main():
    ap = argparse.ArgumentParser(
        description="Fold-conservation, heterodimer-packing, and reference-interface benchmarking for integrin heterodimer PDB models."
    )
    ap.add_argument("--input-dir", required=True, help="Directory containing predicted heterodimer PDB files.")
    ap.add_argument("--output-dir", default="integrin_heterodimer_qc_results", help="Output directory.")
    ap.add_argument("--alpha-chain", default="A", help="Alpha-integrin chain ID in the prediction files.")
    ap.add_argument("--beta-chain", default="B", help="Beta-integrin chain ID in the prediction files.")
    ap.add_argument("--reference", default=None, help="Optional native/reference heterodimer PDB with matching chain IDs and residue numbering.")
    ap.add_argument("--templates-dir", default=None, help="Optional directory of reference/template PDBs for fold conservation.")
    ap.add_argument("--domain-map", default=None, help="Optional CSV with columns: domain,chain,start,end for domain-level fold conservation.")
    ap.add_argument("--skip-plots", action="store_true", help="Do not generate summary PNG plots.")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    pdbs = sorted(input_dir.glob("*.pdb"))
    if not pdbs:
        raise SystemExit(f"No PDB files found in {input_dir}")

    reference_struct = parse_structure(Path(args.reference)) if args.reference else None
    templates = load_templates(Path(args.templates_dir)) if args.templates_dir else {}
    domains = read_domain_map(Path(args.domain_map)) if args.domain_map else []

    packing_rows = []
    fold_rows = []

    for pdb in pdbs:
        structure = parse_structure(pdb)
        row = analyze_packing(pdb, structure, args.alpha_chain, args.beta_chain)
        if reference_struct is not None:
            row.update(benchmark_against_reference(structure, reference_struct, args.alpha_chain, args.beta_chain))
        packing_rows.append(row)

        if templates:
            fold_rows.extend(
                fold_comparison_rows(
                    pdb.name,
                    structure,
                    templates,
                    [args.alpha_chain, args.beta_chain],
                    domains,
                    tmp_dir,
                )
            )

    packing_df = add_reference_scores(add_packing_scores(pd.DataFrame(packing_rows)))
    packing_csv = output_dir / "heterodimer_packing_interface_summary.csv"
    packing_df.to_csv(packing_csv, index=False)

    if fold_rows:
        fold_df = pd.DataFrame(fold_rows)
        # Conservative fold score: prefer template-normalized TM-score when present; keep fallback components visible.
        tm_col = "tm_score_reference_norm"
        if tm_col in fold_df.columns:
            fold_df["Fold_conservation_index"] = pd.to_numeric(fold_df[tm_col], errors="coerce")
            fallback = pd.to_numeric(fold_df["Fallback_CMO"], errors="coerce")
            fold_df["Fold_conservation_index"] = fold_df["Fold_conservation_index"].fillna(fallback)
        else:
            fold_df["Fold_conservation_index"] = pd.to_numeric(fold_df["Fallback_CMO"], errors="coerce")
        fold_df.sort_values(["Model", "Chain", "Fold_unit", "Fold_conservation_index"], ascending=[True, True, True, False], inplace=True)
        fold_csv = output_dir / "fold_conservation_benchmark.csv"
        fold_df.to_csv(fold_csv, index=False)
    else:
        fold_df = pd.DataFrame()

    # Write top contact pairs for each model for manual biological inspection.
    contacts_dir = output_dir / "interface_contact_pairs"
    contacts_dir.mkdir(exist_ok=True)
    for pdb in pdbs:
        structure = parse_structure(pdb)
        _, _, residue_pairs, _ = detect_interface(structure, args.alpha_chain, args.beta_chain)
        pd.DataFrame(sorted(residue_pairs), columns=["Alpha_interface_residue", "Beta_interface_residue"]).to_csv(
            contacts_dir / f"{pdb.stem}_interface_pairs.csv", index=False
        )

    if not args.skip_plots:
        plot_bar(
            packing_df,
            "Packing_quality_index",
            output_dir / "packing_quality_index.png",
            "Integrin heterodimer packing quality index",
        )
        plot_metric_heatmap(
            packing_df,
            [
                "Interface_residues",
                "Residue_pair_contacts",
                "Contact_density",
                "Hbond_density",
                "Salt_bridge_density",
                "Hydrophobic_contact_density",
                "BSA_total_A2",
                "Interface_mean_bfactor_or_plddt",
            ],
            output_dir / "packing_metrics_heatmap.png",
            "Heterodimer interface packing metrics",
        )
        if "Reference_interface_benchmark_index" in packing_df.columns:
            plot_bar(
                packing_df,
                "Reference_interface_benchmark_index",
                output_dir / "reference_interface_benchmark_index.png",
                "Reference-based interface benchmark index",
            )

    print("Wrote:", packing_csv)
    if fold_rows:
        print("Wrote:", output_dir / "fold_conservation_benchmark.csv")
    print("Wrote contact-pair tables to:", contacts_dir)


if __name__ == "__main__":
    main()
