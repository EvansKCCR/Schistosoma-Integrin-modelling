# Integrin AF3 Post-processing Toolkit

## Overview
This Module contains utility scripts for post-processing AlphaFold3 (AF3) structural predictions and structural file handling. The tools support:

- Extraction and summarization of AF3 confidence metrics  
- Conversion of structural files from mmCIF to PDB format  
- Aggregation and reporting of model quality metrics across multiple predictions  

These scripts facilitate downstream structural analysis, docking preparation, and quality assessment.

---

## Input Requirements

### 1. AF3 JSON files
- Generated from AlphaFold3 predictions  
- Contain confidence metrics such as:
  - PAE (Predicted Aligned Error)
  - pLDDT
  - contact probabilities
  - pTM / ipTM scores  

### 2. Structural files (.cif)
- AlphaFold or other structural outputs in mmCIF format

---

## Directory Structure

project_root/
│
├── models/
│   ├── model_1.json
│   ├── model_2.json
│   └── ...
│
├── cif_files/
│   ├── structure1.cif
│   ├── structure2.cif
│   └── ...
│
├── pdb_files/
│
├── scripts/
│   ├── AF3_stats.py
│   ├── confidence_score_summary.py
│   └── batch_cif_to_pdb.py
│
├── results/
│   ├── af3_metrics.csv
│   ├── af3_metrics.xlsx
│   └── ...

---

## File Naming Convention

### AF3 JSON files
model_<ID>.json  
Example: alpha1_beta1_model1.json

### Structure files
structure_<ID>.cif → structure_<ID>.pdb

---

## Installation

Requirements:
- Python ≥ 3.8

Install dependencies:
pip install pandas biopython openpyxl numpy

---

## Usage

### 1. Compute AF3 statistical summaries
python AF3_stats.py models/*.json --out results/af3_metrics.csv

### 2. Generate detailed AF3 Excel report
python confidence_score_summary.py --input_dir models --output results/af3_metrics.xlsx

### 3. Convert CIF → PDB (batch)
python batch_cif_to_pdb.py

---

## Workflow

AF3 prediction → JSON outputs  
→ AF3_stats.py → quick summary  
→ confidence_score_summary.py → detailed Excel  
→ batch_cif_to_pdb.py → structure conversion  

---

## Summary

This toolkit enables:
- Rapid evaluation of AF3 model quality  
- Standardization of structural formats  
- Scalable analysis of multiple models  
