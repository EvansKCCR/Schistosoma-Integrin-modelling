# MD Summary, Plotting, and Residence Time Analysis

## Overview
This repository provides two scripts for post-processing molecular dynamics (MD) trajectories exported to Excel:

1. **generate_summary_and_plots.py**
   - Produces a comprehensive Excel summary and a full set of high-resolution (600 dpi) figures from trajectory data.
   - Includes equilibration detection, RMSD/RMSF/rGyr statistics, surface metrics, and protein–ligand contact visualizations.

2. **residence_time_analysis_script.py**
   - Computes ligand residence time metrics from ligand RMSD traces.
   - Generates residence time comparison plots and survival probability curves.

Together, these scripts enable **quantitative assessment of MD stability and binding behavior** with publication-ready outputs.

---

## Input Requirements

### trajectory_data.xlsx
Both scripts expect an Excel workbook named:
```
trajectory_data.xlsx
```

### Required sheets (generate_summary_and_plots.py)
- C_alpha_RMSD
- Lig_RMSD
- rGyr
- SASA2
- MolSA2
- PSA_
- intraHB
- RMSF
- P_L_contact

### Required sheets (residence_time_analysis_script.py)
- Lig_RMSD

### Data format expectations
- Columns: `Frame`, `Time (ns)`, and one column per condition (e.g., RGD, Mutated, Pos control)
- Numeric values for all metrics (non-numeric entries are coerced)

---

## Directory Structure

```
project_root/
│
├── trajectory_data.xlsx
│
├── scripts/
│   ├── generate_summary_and_plots.py
│   └── residence_time_analysis_script.py
│
├── outputs/
│   ├── MD_summary.xlsx
│   ├── retention_MD_summary.xlsx
│   └── figures/
```

---

## File Naming Convention

### Input
```
trajectory_data.xlsx
```

### Outputs (generate_summary_and_plots.py)
- `MD_summary.xlsx`
- Figures:
  - Fig1_RMSD_timecourse.png
  - Fig2_LigandRMSD_timecourse.png
  - Fig3_LigandRMSD_distributions.png
  - Fig4_rGyr_timecourse.png
  - Fig5_SurfaceMetrics_RGD_vs_Mut.png
  - Fig6_DeltaRMSF.png
  - Fig7–Fig12 protein–ligand contact and proxy plots

### Outputs (residence_time_analysis_script.py)
- `retention_MD_summary.xlsx`
- Figures:
  - Fig13_LigandResidenceTime.png
  - Fig14_ResidenceTimeSurvival.png

---

## Installation

### Requirements
- Python ≥ 3.8

Install dependencies:
```
pip install pandas numpy matplotlib scipy scikit-learn openpyxl
```

---

## Usage

### 1. Generate full MD summary and plots

```
python generate_summary_and_plots.py
```

**Outputs:**
- Comprehensive Excel summary of all MD metrics
- High-resolution publication-ready figures

---

### 2. Compute ligand residence time metrics

```
python residence_time_analysis_script.py
```

**Outputs:**
- Residence time statistics:
  - Mean residence time (τ_bound)
  - Maximum residence time (τ_max)
- Survival probability curves
- Comparative residence time plots

---

## Workflow

```
trajectory_data.xlsx
        ↓
generate_summary_and_plots.py
        ↓
MD_summary.xlsx + Figures (Fig1–Fig12)
        ↓
residence_time_analysis_script.py
        ↓
retention_MD_summary.xlsx + Figures (Fig13–Fig14)
```

---

## Key Features

- Automated equilibration detection using rolling RMSD stability
- Dual-window analysis (full trajectory vs equilibrated region)
- High-resolution (600 dpi) figure generation
- Per-residue and interface-level analysis
- Ligand binding kinetics via residence time and survival curves

---

## Summary

This toolkit provides a **complete MD analysis layer** for:

- Structural stability assessment
- Ligand binding persistence
- Comparative condition analysis (e.g., RGD vs Mutant)
- Publication-quality visualization

It is particularly suited for:
- Protein–ligand interaction studies
- Integrin binding analysis
- Molecular dynamics benchmarking workflows
