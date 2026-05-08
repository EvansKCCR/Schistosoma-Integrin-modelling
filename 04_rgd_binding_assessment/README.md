# Integrin–RGD Binder Ranking Pipeline (v3)

A robust structural scoring pipeline for evaluating and ranking RGD–integrin models using biophysically meaningful features: interface contact density, buried surface area (BSA), hydrogen bonding, salt bridges, MIDAS-site coordination geometry, and ensemble hotspot mapping.

This version extends the original v2 workflow with:

- Advanced MIDAS/Mg²⁺ coordination analysis
- Residue-level contact hotspot mapping
- Ensemble contact-frequency analysis
- Ligand-class interaction profiling
- Improved Mg²⁺ detection heuristics
- Frequency heatmaps and ligand comparison grids
- Enhanced PDB sanitization and MDAnalysis compatibility

---

## What this version includes (v3)

### Core scoring framework
- Composite scoring with configurable weights (contacts, BSA, H-bonds, salt bridges, MIDAS).
- ARG0 penalty: ranking is penalized if no ARG salt bridge is detected.
- RGD–receptor residue contact map generation (CSV + optional PNG heatmap).
- PDB sanitizer: rewrites the PDB element column (cols 77–78) from atom names.
- Optional suppression of MDAnalysis INFO chatter (`--quiet_mda`).

### New v3 features
- Ensemble contact-frequency hotspot analysis
- Ligand-class frequency aggregation
- Binary and frequency hotspot maps
- Grid-based ligand comparison figures
- Improved MIDAS/Mg²⁺ coordination heuristics
- Residue hotspot summaries
- Octahedral geometry heuristic scoring
- Ligand denticity estimation

---

## Repository files / entry points

| File | Description |
|---|---|
| `rank_integrin_binding_v3_hotspots.py` | Main v3 pipeline with hotspot analysis |
| `rank_integrin_binding_v2_quiet.py` | Stable v2 workflow |
| `rank_integrin_binding_v2.py` | Legacy v2 variant |
| `rank_integrin_binding.py` | Original baseline version |

> Recommended: use `rank_integrin_binding_v3_hotspots.py` for full hotspot and ensemble analysis.

---

## Installation

### Create a recommended conda environment

```bash
conda create -n integrin python=3.10
conda activate integrin
conda install -c conda-forge mdanalysis freesasa scipy numpy pandas joblib matplotlib
```

Notes:
- `freesasa` is optional; the pipeline falls back to a proxy BSA if unavailable.
- Python 3.10+ is recommended.

---

## Usage

### Basic run (rank models)

```bash
python rank_integrin_binding_v3_hotspots.py \
  --input_dir pdb_models/ \
  --rgd_chain C \
  --receptor_chains A,B
```

---

### Use FreeSASA for real BSA

```bash
python rank_integrin_binding_v3_hotspots.py \
  --input_dir pdb_models/ \
  --rgd_chain C \
  --receptor_chains A,B \
  --use_freesasa
```

---

### Fast contacts + bidirectional H-bonds

```bash
python rank_integrin_binding_v3_hotspots.py \
  --input_dir pdb_models/ \
  --rgd_chain C \
  --receptor_chains A,B \
  --use_freesasa \
  --fast_contacts \
  --hbonds_bidirectional
```

---

### Export Top-N + plots + report

```bash
python rank_integrin_binding_v3_hotspots.py \
  --input_dir pdb_models/ \
  --rgd_chain C \
  --receptor_chains A,B \
  --use_freesasa \
  --top_n 10 \
  --plots \
  --report
```

---

## Contact maps and hotspot analysis

### Generate contact maps

```bash
python rank_integrin_binding_v3_hotspots.py \
  --input_dir pdb_models/ \
  --rgd_chain C \
  --receptor_chains A,B \
  --contact_map all \
  --plots
```

### Binary hotspot maps

```bash
python rank_integrin_binding_v3_hotspots.py \
  --input_dir pdb_models/ \
  --rgd_chain C \
  --receptor_chains A,B \
  --contact_map all \
  --contact_map_binary \
  --plots
```

---

## Ensemble contact-frequency analysis (new in v3)

### Aggregate hotspot frequencies across models

```bash
python rank_integrin_binding_v3_hotspots.py \
  --input_dir pdb_models/ \
  --rgd_chain C \
  --receptor_chains A,B \
  --contact_map all \
  --contact_frequency grid \
  --plots
```

### Use only top-ranked models

```bash
python rank_integrin_binding_v3_hotspots.py \
  --input_dir pdb_models/ \
  --rgd_chain C \
  --receptor_chains A,B \
  --contact_frequency grid \
  --freq_models top \
  --freq_top_n 50 \
  --plots
```

### Customize hotspot threshold

```bash
--freq_threshold 0.5
```

This retains contacts present in >50% of models.

---

## Ligand-class frequency grids

Built-in ligand recognition supports:

- GRGDSP
- GRGESP
- RGD4C
- iRGD
- VP7_nonRGD
- GFOGER
- IKVAV
- YIGSR
- GYRGDGQ

Generate ligand-comparison grids:

```bash
python rank_integrin_binding_v3_hotspots.py \
  --input_dir pdb_models/ \
  --rgd_chain C \
  --receptor_chains A,B \
  --contact_frequency grid \
  --plots
```

---

## Scoring controls

### Override scoring weights

```bash
--weights "bsa=0.01,salt=2,hbond=1.5,contacts=0.1,midas=3"
```

### Customize MIDAS optimal/acceptable windows

```bash
--midas_opt 2.0 2.2 --midas_acc 2.2 2.6
```

### ARG0 penalty

```bash
--arg0_penalty 5.0
```

Disable:

```bash
--arg0_penalty 0
```

---

## New MIDAS diagnostics (v3)

| Metric | Meaning |
|---|---|
| `AspMg_min_dist` | Minimum acidic oxygen → Mg²⁺ distance |
| `Mg_coord_count` | Number of Mg²⁺ coordinating atoms |
| `Mg_coord_var` | Variance of coordination distances |
| `AcidMg_n_inner` | Ligand acidic oxygens within shell |
| `Donor_count_3A` | Donor atoms within 3 Å |
| `MgO_mean` | Mean Mg–donor distance |
| `MgO_std` | Mg coordination standard deviation |
| `Octahedral_heuristic` | Approximate octahedral geometry quality |
| `Ligand_denticity` | Mono/bidentate ligand estimate |

---

## Output files

| File | Description |
|---|---|
| `RGD_binder_ranking.csv` | Full ranking of all models |
| `RGD_top<N>.csv` | Optional Top-N subset |
| `RGD_top_report.txt` | Top-model summary |
| `plots/*.png` | Scatterplots and barplots |
| `contact_maps/*.csv` | Contact matrices |
| `contact_maps/*.png` | Contact heatmaps |
| `contact_frequency_grid_*.png` | Ensemble ligand grids |
| `*_fixed.pdb` | Sanitized structures |

---

## Key ranking columns

| Column | Meaning |
|---|---|
| `Biological_score` | Base composite score |
| `Rank_score` | Final penalized score |
| `Classification` | ARG salt-bridge classification |
| `contacts` | Heavy-atom contact count |
| `BSA` | Buried surface area |
| `salt_bridges` | Total salt bridges |
| `ARG_salt_pairs` | ARG-mediated salt bridges |
| `H_bonds` | Hydrogen-bond count |
| `MIDAS_score` | MIDAS coordination score |
| `Octahedral_heuristic` | Mg²⁺ geometry quality |
| `Ligand_denticity` | Ligand coordination estimate |

---

## Composite score

### Biological_score

```text
Biological_score =
    (w_bsa      * BSA) +
    (w_salt     * salt_bridges) +
    (w_hbond    * H_bonds) +
    (w_contacts * contacts) +
    (w_midas    * MIDAS_score)
```

### Rank_score (penalized)

```text
Rank_score = Biological_score - ARG0_penalty
```

if:

```text
ARG_salt_residues == 0
```

---

## Recommended “quiet + clean” run

```bash
python rank_integrin_binding_v3_hotspots.py \
  --input_dir pdb_models/ \
  --rgd_chain C \
  --receptor_chains A,B \
  --sanitize_pdb \
  --quiet_mda \
  --use_freesasa \
  --contact_map all \
  --contact_frequency grid \
  --contact_map_binary \
  --freq_models top \
  --freq_top_n 50 \
  --freq_threshold 0.5 \
  --plots \
  --report \
  --log WARNING
```

---

## Biological applications

This pipeline is suitable for:

- Integrin–ligand docking analysis
- RGD peptide screening
- Adhesome reconstruction
- MIDAS-site geometry analysis
- ECM–integrin interaction mapping
- Comparative ligand hotspot analysis
- Parasite adhesome studies
- Multivalent binding characterization

---

## Troubleshooting

### Unknown element warnings

Run with:

```bash
--sanitize_pdb
```

### MDAnalysis topology chatter

Use:

```bash
--quiet_mda --log WARNING
```

### FreeSASA unavailable

Install:

```bash
conda install -c conda-forge freesasa
```

### MIDAS diagnostics appear incorrect

Provide:

```bash
--mg_chain <ID>
```

---

## Recommended visualization tools

- PyMOL
- ChimeraX
- VMD
- Cytoscape
- GraphPad Prism
- Illustrator/Inkscape

---

## Citation

If you use this pipeline in published work, please cite:

- MDAnalysis (https://pmc.ncbi.nlm.nih.gov/articles/PMC3144279/)
- FreeSASA (https://pmc.ncbi.nlm.nih.gov/articles/PMC4776673/)
- Upstream docking software
- Associated integrin adhesome analyses

---

## Notes

- Heavy-atom contact calculations ignore hydrogens when `--heavy_only` is enabled.
- Frequency maps represent ensemble interaction persistence.
- Hotspot maps retain only non-zero contacts.
- MIDAS heuristics are optimized for Mg²⁺-dependent integrin systems.
- Ensemble grids facilitate comparative ligand-binding analysis.

---

## Example interpretation

High-quality binders typically exhibit:

- High BSA
- Strong ARG-mediated salt bridges
- Stable MIDAS coordination
- Low Mg²⁺ coordination variance
- Persistent ensemble hotspots
- Multivalent interaction patterns

These features are consistent with biologically stable integrin–ligand engagement.

