## Integrin–RGD Binder Ranking Pipeline (v2)

A robust structural scoring pipeline for evaluating and ranking RGD–integrin models using biophysically meaningful features: interface contact density, buried surface area (BSA), hydrogen bonding, salt bridges, and MIDAS-site coordination geometry.

### What this version includes (v2)
- Composite scoring with configurable weights (contacts, BSA, H-bonds, salt bridges, MIDAS).
- **ARG0 penalty**: ranking is penalized if **no ARG salt bridge** is detected (RGD ARG → receptor ASP/GLU).
- **RGD–receptor residue contact map** generation (CSV + optional PNG heatmap).
- **PDB sanitizer**: rewrites the PDB **element column (cols 77–78)** from atom names and saves `*_fixed.pdb` files. (This helps eliminate “unknown element” parser warnings.)
- Optional suppression of MDAnalysis “guesser/topology” INFO chatter (`--quiet_mda`).

---

## Repository files / entry points

- `rank_integrin_binding_v2_quiet.py` (**recommended**): full v2 feature set (contact maps + ARG0 penalty + sanitizer + quiet MDAnalysis).
- `rank_integrin_binding_v2.py`: v2 pipeline variant (if present in your folder).
- `rank_integrin_binding.py`: legacy baseline script (no contact maps / no ARG0 penalty / no sanitizer).

> Tip: Prefer `rank_integrin_binding_v2_quiet.py` for the cleanest CLI + logging behavior.

---

## Installation

### 1) Create a recommended conda environment
```bash
conda create -n integrin python=3.10
conda activate integrin
conda install -c conda-forge mdanalysis freesasa scipy numpy pandas joblib matplotlib
```

Notes:
- `freesasa` is optional; if unavailable, the pipeline automatically falls back to a BSA proxy.
- Python 3.10+ is recommended for MDAnalysis + ecosystem stability.

---

## Usage

### Basic run (rank models)
```bash
python rank_integrin_binding_v2_quiet.py \
  --input_dir pdb_models/ \
  --rgd_chain C \
  --receptor_chains A,B
```

### Use FreeSASA for real BSA (if installed)
```bash
python rank_integrin_binding_v2_quiet.py \
  --input_dir pdb_models/ \
  --rgd_chain C \
  --receptor_chains A,B \
  --use_freesasa
```

### Fast contacts + bidirectional H-bonds
```bash
python rank_integrin_binding_v2_quiet.py \
  --input_dir pdb_models/ \
  --rgd_chain C \
  --receptor_chains A,B \
  --use_freesasa \
  --fast_contacts \
  --hbonds_bidirectional
```

### Export Top-N + plots + report
```bash
python rank_integrin_binding_v2_quiet.py \
  --input_dir pdb_models/ \
  --rgd_chain C \
  --receptor_chains A,B \
  --use_freesasa \
  --top_n 10 \
  --plots \
  --report
```

### Generate contact maps

**Top model only**
```bash
python rank_integrin_binding_v2_quiet.py \
  --input_dir pdb_models/ \
  --rgd_chain C \
  --receptor_chains A,B \
  --contact_map top \
  --plots
```

**All models + aggregate frequency map**
```bash
python rank_integrin_binding_v2_quiet.py \
  --input_dir pdb_models/ \
  --rgd_chain C \
  --receptor_chains A,B \
  --contact_map all \
  --plots
```

**Binary contact map (0/1 instead of counts)**
```bash
python rank_integrin_binding_v2_quiet.py \
  --input_dir pdb_models/ \
  --rgd_chain C \
  --receptor_chains A,B \
  --contact_map all \
  --contact_map_binary \
  --plots
```

### Enable PDB element-column sanitizer (writes *_fixed.pdb and analyzes those)
```bash
python rank_integrin_binding_v2_quiet.py \
  --input_dir pdb_models/ \
  --rgd_chain C \
  --receptor_chains A,B \
  --sanitize_pdb
```

### Quiet MDAnalysis INFO chatter (guessers/topology)
```bash
python rank_integrin_binding_v2_quiet.py \
  --input_dir pdb_models/ \
  --rgd_chain C \
  --receptor_chains A,B \
  --sanitize_pdb \
  --quiet_mda \
  --log WARNING
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

### ARG0 penalty (penalize if no ARG salt bridge)
Default penalty is applied when `ARG_salt_residues == 0`. You can adjust or disable it:
```bash
--arg0_penalty 5.0   # default
--arg0_penalty 0     # disable penalty
```

---

## Output files

| File | Description |
|---|---|
| `RGD_binder_ranking.csv` | Full ranking of all models (sorted by `Rank_score`) |
| `RGD_top<N>.csv` | (Optional) Top‑N subset when `--top_n N` is used |
| `RGD_top_report.txt` | (Optional) Text summary of top model when `--report` is used |
| `plots/*.png` | (Optional) Scatterplots + Top‑N bar chart when `--plots` is used |
| `contact_maps/contact_map_top.csv` | (Optional) Contact map for top model (`--contact_map top`) |
| `contact_maps/contact_map_<model>.csv` | (Optional) Per-model maps (`--contact_map all`) |
| `contact_maps/contact_map_frequency.csv` | (Optional) Aggregate contact frequency map (`--contact_map all`) |
| `contact_maps/*.png` | (Optional) Heatmaps (only written if `--plots` is enabled) |
| `*_fixed.pdb` | (Optional) Sanitized PDBs if `--sanitize_pdb` is enabled |

### Key columns in `RGD_binder_ranking.csv`
- `Biological_score`: base composite score (no ARG0 penalty)
- `Rank_score`: final ranking score (includes ARG0 penalty, if triggered)
- `Classification`: `ARG_salt_present` or `penalized_no_ARG_salt`
- `ARG_salt_pairs`, `ARG_salt_residues`, `ARG0_penalty`
- `AspMg_min_dist`, `Mg_coord_count`, `Mg_coord_var` (MIDAS diagnostics)

---

## Metrics computed

- **Interface Contacts**: heavy-atom contacts within `--contact_cutoff`
- **Buried Surface Area (BSA)**: FreeSASA if available; proxy otherwise
- **Salt Bridges**: RGD (ARG/LYS) ↔ receptor (ASP/GLU)
- **ARG-only salt bridges**: RGD ARG ↔ receptor (ASP/GLU) (used for penalty)
- **Hydrogen Bonds**: distance-only heuristic
- **MIDAS-site score**: Asp–Mg distance category, coordination count, compactness/variance + penalties

---

## Composite score

### Biological_score
```
Biological_score =
    (w_bsa      * BSA) +
    (w_salt     * salt_bridges) +
    (w_hbond    * H_bonds) +
    (w_contacts * contacts) +
    (w_midas    * MIDAS_score)
```

### Rank_score (penalized)
If `ARG_salt_residues == 0`:
```
Rank_score = Biological_score - arg0_penalty
```

---

## Troubleshooting

### “Unknown element …” / element parsing warnings
- Run with `--sanitize_pdb` to create `*_fixed.pdb` files and analyze those.
- The sanitizer rewrites the **PDB element field (cols 77–78)** from atom names.

### MDAnalysis messages about `to_guess` only filling empty values
- MDAnalysis distinguishes `to_guess` (fill empties) vs `force_guess` (overwrite). This pipeline uses overwrite for `elements`.
- For cleaner logs, run `--quiet_mda --log WARNING`.

### FreeSASA fails
- Ensure FreeSASA is installed:
  `conda install -c conda-forge freesasa`
- The pipeline automatically falls back to a proxy BSA if FreeSASA is missing.

### Mg²⁺ not detected / MIDAS diagnostics look wrong
- Provide a hint via `--mg_chain <ID>`
- Inspect diagnostics: `AspMg_min_dist`, `Mg_coord_count`, `Mg_coord_var`

---

## Recommended “quiet + clean” run
```bash
python rank_integrin_binding_v2_quiet.py \
  --input_dir pdb_models/ \
  --rgd_chain C \
  --receptor_chains A,B \
  --sanitize_pdb \
  --quiet_mda \
  --contact_map all \
  --contact_map_binary \
  --plots \
  --log WARNING
```
