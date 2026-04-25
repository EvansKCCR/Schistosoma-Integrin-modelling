#!/usr/bin/env python3
"""
Interface benchmarking

Notes:
- Joint A+B Cα alignment
- Interface-only RMSD
- True BSA (selection-free)
- Efficient NeighborSearch interface detection
- Residue-level contact uniqueness
- Improved scoring & Pareto optimization

Script:

interface_benchmark_publication.py

Input:
Folder of predicted complexes (.pdb)
Native/reference structure
Chain IDs
Run:
python interface_benchmark_publication.py \
    predictions/ native.pdb A B
Output:
publication_interface_benchmark.csv
What it computes:
Contact, H-bond, salt bridge densities
Buried surface area (BSA)
Global + interface RMSD
Structural plausibility index
Pareto-optimal models

"""

import os
import sys
import glob
import math
import argparse
import tempfile
import numpy as np
import pandas as pd
from typing import Set, Tuple, List, Dict
from Bio.PDB import PDBParser, Superimposer, NeighborSearch, PDBIO, Select
from Bio.Align import PairwiseAligner
from Bio.Data.IUPACData import protein_letters_3to1

CONTACT_CUTOFF = 5.0
HBOND_DIST = 3.5
SALT_DIST = 4.0
HYDROPHOBIC_DIST = 4.5
MIN_CA_PAIRS = 10

HYDROPHOBIC_3 = {"ALA","VAL","ILE","LEU","MET","PHE","TYR","TRP","PRO","CYS"}
VDW_RADII = {"C":1.7,"N":1.55,"O":1.52,"S":1.8,"P":1.8}
DEFAULT_VDW = 1.7

# ---------------------------
# Utilities
# ---------------------------

def is_standard_res(res):
    return res.get_id()[0] == " "

def three_to_one_safe(resname):
    return protein_letters_3to1.get(resname.capitalize(),"X")

def get_structure(path):
    return PDBParser(QUIET=True).get_structure("s", path)

def zscore(series, invert=False):
    s = pd.to_numeric(series, errors="coerce")
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or math.isnan(sd):
        z = pd.Series(0.0, index=s.index)
    else:
        z = (s - mu) / sd
    return -z if invert else z

# ---------------------------
# Joint Alignment (A+B)
# ---------------------------

def joint_align(pred, native, chainA, chainB):
    pred_atoms, nat_atoms = [], []
    for chain in [chainA, chainB]:
        if chain not in pred[0] or chain not in native[0]:
            continue
        for res in pred[0][chain]:
            if not is_standard_res(res) or "CA" not in res:
                continue
            rid = res.get_id()
            if rid in native[0][chain] and "CA" in native[0][chain][rid]:
                pred_atoms.append(res["CA"])
                nat_atoms.append(native[0][chain][rid]["CA"])
    if len(pred_atoms) < MIN_CA_PAIRS:
        return np.nan
    sup = Superimposer()
    sup.set_atoms(nat_atoms, pred_atoms)
    sup.apply(pred.get_atoms())
    return sup.rms

# ---------------------------
# Interface detection (NeighborSearch)
# ---------------------------

def get_interface(structure, chainA, chainB):
    atomsA = [a for a in structure[0][chainA].get_atoms() if a.element!="H"]
    atomsB = [a for a in structure[0][chainB].get_atoms() if a.element!="H"]
    ns = NeighborSearch(atomsA + atomsB)

    resA_set, resB_set = set(), set()
    residue_pairs = set()

    for atom in atomsA:
        neighbors = ns.search(atom.coord, CONTACT_CUTOFF)
        for n in neighbors:
            if n.get_parent().get_parent().id != chainB:
                continue
            resA = atom.get_parent()
            resB = n.get_parent()
            resA_set.add(resA)
            resB_set.add(resB)
            residue_pairs.add((resA.get_id(), resB.get_id()))

    return resA_set, resB_set, residue_pairs

# ---------------------------
# Interface-only RMSD
# ---------------------------

def interface_rmsd(pred, native, chainA, chainB, resA_set, resB_set):
    pred_atoms, nat_atoms = [], []

    for res in resA_set:
        rid = res.get_id()
        if rid in native[0][chainA] and "CA" in res:
            pred_atoms.append(res["CA"])
            nat_atoms.append(native[0][chainA][rid]["CA"])

    for res in resB_set:
        rid = res.get_id()
        if rid in native[0][chainB] and "CA" in res:
            pred_atoms.append(res["CA"])
            nat_atoms.append(native[0][chainB][rid]["CA"])

    if len(pred_atoms) < 5:
        return np.nan

    sup = Superimposer()
    sup.set_atoms(nat_atoms, pred_atoms)
    return sup.rms

# ---------------------------
# H-bonds (proxy)
# ---------------------------

def count_hbonds(structure, residue_pairs, chainA, chainB):
    count = 0
    m = structure[0]
    for ridA, ridB in residue_pairs:
        resA = m[chainA][ridA]
        resB = m[chainB][ridB]
        for a in resA:
            if a.element not in ("N","O"): continue
            for b in resB:
                if b.element not in ("N","O"): continue
                if a - b <= HBOND_DIST:
                    count += 1
    return count

# ---------------------------
# Salt bridges
# ---------------------------

def count_salt_bridges(structure, chainA, chainB):
    charged = {"ASP","GLU","LYS","ARG"}
    pairs = set()
    m = structure[0]
    for resA in m[chainA]:
        if resA.get_resname() not in charged: continue
        for resB in m[chainB]:
            if resB.get_resname() not in charged: continue
            for a in resA:
                for b in resB:
                    if a - b <= SALT_DIST:
                        pairs.add((resA.get_id(), resB.get_id()))
    return len(pairs)

# ---------------------------
# True BSA
# ---------------------------

class ChainSelect(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id
    def accept_chain(self, chain):
        return chain.id == self.chain_id
    def accept_residue(self, residue):
        return is_standard_res(residue)
    def accept_atom(self, atom):
        return atom.element != "H"

def compute_true_bsa(pdb_path, structure, chainA, chainB):
    import freesasa
    complex_area = freesasa.calc(freesasa.Structure(pdb_path)).totalArea()

    io = PDBIO()
    io.set_structure(structure)

    with tempfile.TemporaryDirectory() as td:
        pa = os.path.join(td,"A.pdb")
        pb = os.path.join(td,"B.pdb")
        io.save(pa, ChainSelect(chainA))
        io.save(pb, ChainSelect(chainB))
        asaA = freesasa.calc(freesasa.Structure(pa)).totalArea()
        asaB = freesasa.calc(freesasa.Structure(pb)).totalArea()

    return asaA + asaB - complex_area

# ---------------------------
# Pareto
# ---------------------------

def compute_pareto(df, cols):
    X = df[cols].fillna(-np.inf).to_numpy()
    pareto = []
    for i in range(len(X)):
        dominated = False
        for j in range(len(X)):
            if i==j: continue
            if (X[j]>=X[i]).all() and (X[j]>X[i]).any():
                dominated=True
                break
        pareto.append(not dominated)
    return pareto

# ---------------------------
# Main
# ---------------------------

def main():

    if len(sys.argv)<5:
        print("Usage: python v3_2.py predictions_folder native.pdb A B")
        sys.exit(1)

    pred_folder = sys.argv[1]
    native_file = sys.argv[2]
    chainA = sys.argv[3]
    chainB = sys.argv[4]

    native = get_structure(native_file)

    rows = []

    for pdb_file in glob.glob(os.path.join(pred_folder,"*.pdb")):

        pred = get_structure(pdb_file)

        global_rmsd = joint_align(pred, native, chainA, chainB)

        resA_set, resB_set, residue_pairs = get_interface(pred, chainA, chainB)
        iface_size = len(resA_set) + len(resB_set)

        iface_rmsd = interface_rmsd(pred, native, chainA, chainB, resA_set, resB_set)

        hbonds = count_hbonds(pred, residue_pairs, chainA, chainB)
        salts = count_salt_bridges(pred, chainA, chainB)

        contact_density = len(residue_pairs) / max(iface_size,1)
        hbond_density = hbonds / max(iface_size,1)
        salt_density = salts / max(iface_size,1)

        try:
            bsa = compute_true_bsa(pdb_file, pred, chainA, chainB)
        except:
            bsa = np.nan

        rows.append({
            "Model": os.path.basename(pdb_file),
            "Interface_residues": iface_size,
            "Contact_density": contact_density,
            "Hbond_density": hbond_density,
            "Salt_density": salt_density,
            "BSA_total": bsa,
            "Global_RMSD_AB": global_rmsd,
            "Interface_RMSD": iface_rmsd
        })

    df = pd.DataFrame(rows)

    # Z-scores
    df["Contact_density_z"] = zscore(df["Contact_density"])
    df["Hbond_density_z"] = zscore(df["Hbond_density"])
    df["Salt_density_z"] = zscore(df["Salt_density"])
    df["BSA_total_z"] = zscore(df["BSA_total"])
    df["Global_RMSD_z"] = zscore(df["Global_RMSD_AB"], invert=True)
    df["Interface_RMSD_z"] = zscore(df["Interface_RMSD"], invert=True)

    zcols = [c for c in df.columns if c.endswith("_z")]

    # Weighted plausibility index
    weights = {
        "Contact_density_z":0.20,
        "Hbond_density_z":0.20,
        "Salt_density_z":0.15,
        "BSA_total_z":0.15,
        "Global_RMSD_z":0.15,
        "Interface_RMSD_z":0.15
    }

    df["Structural_plausibility_index"] = sum(df[c]*w for c,w in weights.items())

    pareto_cols = ["Contact_density","Hbond_density","Salt_density","BSA_total"]
    df["Pareto_optimal"] = compute_pareto(df, pareto_cols)

    df.sort_values("Structural_plausibility_index", ascending=False, inplace=True)

    df.to_csv("publication_interface_benchmark.csv", index=False)

    print("\nMaximal robustness benchmarking complete.")
    print("Saved to publication_interface_benchmark.csv\n")

if __name__ == "__main__":
    main()