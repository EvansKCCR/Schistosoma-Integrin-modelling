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
| `integrin_rgd_qc_benchmark_pipeline.py` | Integrated integrin-RGD complex QC/benchmarking orchestrator that runs ligand ranking, heterodimer packing QC, stereochemical/Voronoi QC, chain auto-detection, and merged reporting |
| `rank_integrin_binding_v3_hotspots.py` | Main v3 pipeline with hotspot analysis |

> Recommended: use `integrin_rgd_qc_benchmark_pipeline.py` for complete complex QC and benchmarking, or `rank_integrin_binding_v3_hotspots.py` when you only need ligand-interface ranking/hotspot analysis.

---

## Integrated QC and benchmarking pipeline

The integrated entry point combines the three scripts in this directory into a single workflow:

1. Discover PDB files from a directory or individual file paths.
2. Auto-detect alpha-integrin, beta-integrin, RGD peptide, and Mg chains.
3. Stage models by compatible chain groups.
4. Run RGD ligand-interface ranking and contact-frequency hotspot analysis.
5. Run heterodimer packing/interface QC.
6. Run stereochemical, backbone, and VoroMQA/Voronoi QC when dependencies/tools are available.
7. Merge outputs into one model-level CSV.

### Inspect chain detection only

```bash
python RGD_peptide_binding_benchmarking/integrin_rgd_qc_benchmark_pipeline.py \
  --input-pdb baseline_models/AF3_sm_alpha1_beta1_rgd_model_3.pdb \
  --output-dir AF3_output_analysis/integrated_smoke \
  --list-models
```

For `AF3_sm_alpha1_beta1_rgd_model_3.pdb`, the expected auto-detection is:

```text
alpha=A beta=B rgd=C mg=B
```

### Preview commands without running scientific dependencies

```bash
python RGD_peptide_binding_benchmarking/integrin_rgd_qc_benchmark_pipeline.py \
  --input-pdb baseline_models/AF3_sm_alpha1_beta1_rgd_model_3.pdb \
  --output-dir AF3_output_analysis/integrated_smoke_dry \
  --dry-run \
  --plots \
  --report
```

### Run all baseline models

```bash
python RGD_peptide_binding_benchmarking/integrin_rgd_qc_benchmark_pipeline.py \
  --input-dir baseline_models \
  --pattern "*.pdb" \
  --output-dir AF3_output_analysis/integrin_rgd_qc_benchmark \
  --contact-map all \
  --contact-frequency grid \
  --resume \
  --progress-interval 30 \
  --plots \
  --report \
  --n-jobs 4 \
  --nproc 4
```

### Useful overrides

```bash
--alpha-chain A --beta-chain B --rgd-chain C --mg-chain B
--reference baseline_models/4wk2_xray.pdb
--templates-dir baseline_models
--domain-map domains.csv
--skip-ligand
--skip-packing
--skip-stereo
--resume
--progress-interval 10
--no-progress
```

### Monitoring long runs

The integrated pipeline prints progress bars for model discovery, each chain-compatible group, each sub-pipeline stage, and final merging. Long-running child scripts are monitored with elapsed-time heartbeat messages such as:

```text
ligand group 1/2 still running after 2m30s; last log: ...
```

Each sub-pipeline also writes a full log beside its staged group:

| Log | Description |
|---|---|
| `ligand_binding/*/ligand_benchmark.log` | RGD ranking and hotspot analysis log |
| `heterodimer_packing/*/heterodimer_packing.log` | Heterodimer packing QC log |
| `stereochemical_voronoi/*/stereochemical_voronoi.log` | Stereochemical/backbone/Voronoi QC log |

Use `--resume` after an interrupted or failed run to reuse existing non-empty summary CSVs and continue remaining stages. Use `--require-all` when you want missing dependencies to stop the run immediately instead of skipping unavailable analyses.

If an older run produced `stereo_complex_molprobity_error` values containing `Conflicting scattering type symbols` for calcium records such as `HETATM ... CA   CA`, rerun the stereochemical/Voronoi stage after this cleanup fix without reusing the old stereo summary. The cleaner now rewrites PDB element columns before MolProbity sees the files.

### Integrated output files

| File | Description |
|---|---|
| `model_chain_manifest.csv` | Chain detection and model provenance table |
| `dependency_report.json` | Python module and external-tool availability |
| `combined_integrin_rgd_benchmark_summary.csv` | Merged model-level summary across available sub-pipelines |
| `run_summary.json` | Run metadata and paths to sub-pipeline outputs |
| `ligand_binding/*/run/RGD_binder_ranking.csv` | Per-chain-group RGD ranking output |
| `heterodimer_packing/*/run/packing_qc/heterodimer_packing_interface_summary.csv` | Per-chain-group packing QC output |
| `stereochemical_voronoi/*/run/stereochemical_voronoi_qc/heterodimer_validation_summary.csv` | Per-chain-group stereochemical/Voronoi QC output |

The combined `integrated_score` is the mean of available sub-pipeline summary scores:

```text
ligand Rank_score
packing Packing_quality_index
stereo heterodimer_validation_index
```

If one sub-pipeline is skipped or unavailable, the integrated score is computed from the remaining available scores.

---

## Installation

### Create a recommended conda environment

```bash
conda create -n integrin python=3.10
conda activate integrin
conda install -c conda-forge biopython mdanalysis freesasa scipy numpy pandas joblib matplotlib
```

Notes:
- `freesasa` is optional; the pipeline falls back to a proxy BSA if unavailable.
- `biopython` is required for heterodimer packing and stereochemical/backbone QC.
- MolProbity, CaBLAM, VoroMQA, and TMalign are optional external tools. The integrated pipeline records their availability in `dependency_report.json` and keeps running with the analyses that are available.
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

- MDAnalysis
- FreeSASA
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
