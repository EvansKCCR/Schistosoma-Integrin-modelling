# Integrin Structural Evaluation and Ranking Pipeline

## Overview

This Modules contains a structural evaluation and ranking pipeline for protein–protein complexes, particularly suited for integrin–ligand docking workflows.

The pipeline integrates:

- Interface benchmarking of predicted complexes  
- Bayesian multi-feature ranking of docking models  
- Visualization and sensitivity analysis of scoring outputs  

---

## Input Requirements

### 1. Predicted complex structures (.pdb)

Used by:
- interface_benchmark_publication.py

Requirements:
- Folder containing predicted models (*.pdb)
- A reference/native structure (.pdb)
- Chain identifiers (e.g., A, B)

---

### 2. Benchmarking output

Used by:
- cluspro_mc_bayesian.py

Input file:
classification_matrices.xlsx

Must contain:
- Contact density  
- Hydrogen bond density  
- Salt bridge density  
- Interface identity  
- Global RMSD  
- Predicted binding affinity  

---

### 3. Composite scoring output

Used by:
- generate_plots.py

Input file:
composite_scores_with_sensitivity.xlsx

Required sheets:
- scenario_scores_ranks
- normalized_metrics
- original_input

Optional:
- sensitivity_OAT
- sensitivity_MC_topN

---

## Directory Structure

project_root/
│
├── predictions/
│   ├── model1.pdb
│   ├── model2.pdb
│   └── ...
│
├── native/
│   └── reference.pdb
│
├── results/
│   ├── publication_interface_benchmark.csv
│   ├── classification_matrices.xlsx
│   ├── bayesian_mc_results.xlsx
│   ├── composite_scores_with_sensitivity.xlsx
│   └── plots/
│
├── scripts/
│   ├── interface_benchmark_publication.py
│   ├── cluspro_mc_bayesian.py
│   └── generate_plots.py

---

## File Naming Convention

Predicted structures:
model_<ID>.pdb

Example:
model_01.pdb

Outputs:
publication_interface_benchmark.csv  
classification_matrices.xlsx  
bayesian_mc_results.xlsx  
plots/*.png  

---

## Installation

Requirements:
- Python ≥ 3.8

Install dependencies:

pip install pandas numpy matplotlib seaborn biopython freesasa

---

## Usage

### 1. Interface Benchmarking

python interface_benchmark_publication.py predictions/ native/reference.pdb A B

Output:
publication_interface_benchmark.csv

---

### 2. Bayesian Ranking

python cluspro_mc_bayesian.py

Outputs:
- bayesian_mc_results.xlsx  
- bayesian_stabilization_posterior.png  

---

### 3. Plot Generation

python generate_plots.py composite_scores_with_sensitivity.xlsx

Outputs:
Plots saved in ./plots/

---

## Workflow

Predicted complexes (.pdb)
        ↓
interface_benchmark_publication.py
        ↓
publication_interface_benchmark.csv
        ↓
classification_matrices.xlsx
        ↓
cluspro_mc_bayesian.py
        ↓
bayesian_mc_results.xlsx
        ↓
composite_scores_with_sensitivity.xlsx
        ↓
generate_plots.py
        ↓
Final plots (./plots/)

---

## Summary

This pipeline enables:

- Quantitative evaluation of docking interfaces  
- Robust Bayesian ranking of candidate complexes  
- Visualization of model performance and sensitivity  

It is particularly suited for integrin–ligand interaction studies and protein–protein docking validation.
