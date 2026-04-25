# Integrin–RGD Binder Ranking Pipeline

A robust structural scoring pipeline for evaluating and ranking RGD–integrin models using biophysically meaningful features: contact density, buried surface area (BSA), hydrogen bonding, salt bridges, and MIDAS-site coordination geometry.

This version includes:
- Reliable FreeSASA BSA via MDAnalysis PDBWriter
- Robust Mg²⁺ auto-detection with diagnostics
- Portable chain selection (chainID/segid fallback)
- Adjustable scoring weights
- Configurable MIDAS distance windows
- Optional Top-N export, plots, and text reports
- Fast neighbor-based contact calculation
- Clean error handling and logging

---

## Installation

### **1. Create a recommended environment**
```bash
conda create -n integrin python=3.10
conda activate integrin
conda install -c conda-forge mdanalysis freesasa scipy numpy pandas joblib matplotlib
```

---

## Usage

### **Basic run**
```bash
python rank_integrin_binding.py   --input_dir pdb_models/   --rgd_chain C   --receptor_chains A,B
```

### **Use FreeSASA for real BSA**
```bash
python rank_integrin_binding.py   --input_dir pdb_models/   --rgd_chain C   --receptor_chains A,B   --use_freesasa
```

### **Enable faster neighbor-search and bidirectional H-bonds**
```bash
python rank_integrin_binding.py   --input_dir pdb_models/   --rgd_chain C   --receptor_chains A,B   --use_freesasa   --fast_contacts   --hbonds_bidirectional
```

### **Export Top-10 models + plots + report**
```bash
python rank_integrin_binding.py   --input_dir pdb_models/   --rgd_chain C   --receptor_chains A,B   --use_freesasa   --top_n 10   --plots   --report
```

### **Override scoring weights**
```bash
--weights "bsa=0.01,salt=2,hbond=1.5,contacts=0.1,midas=3"
```

### **Customize MIDAS optimal/acceptable windows**
```bash
--midas_opt 2.0 2.2 --midas_acc 2.2 2.6
```

---

## Output Files

| File | Description |
|------|-------------|
| `RGD_binder_ranking.csv` | Full ranking of all models |
| `RGD_topN.csv` | (Optional) Top‑N subset |
| `RGD_top_report.txt` | (Optional) Text summary of top model |
| `plots/*.png` | (Optional) Scatterplots + Top‑N bar chart |

---

## Metrics Computed

- **Interface Contacts** (heavy atom distance < cutoff)
- **Buried Surface Area (BSA)**    - FreeSASA if available    - Proxy BSA otherwise
- **Salt Bridges**
- **Hydrogen Bonds**
- **MIDAS-site Score**    - Asp–Mg distance category    - Coordination count    - Coordination compactness    - Penalties for under-/over-coordination    - Full diagnostics exported

---

## Composite Score

The “Biological Score” is computed as:

```
score = (w_BSA * BSA)
      + (w_salt * salt_bridges)
      + (w_hbond * H_bonds)
      + (w_contacts * contacts)
      + (w_midas * MIDAS_score)
```

All weights are user‑configurable at runtime.

---

## Troubleshooting

### **FreeSASA fails**
- Ensure FreeSASA is installed (`conda install -c conda-forge freesasa`).
- The script automatically switches to BSA-proxy fallback when needed.

### **Mg²⁺ not detected**
- Provide a hint via:    `--mg_chain I`  - Or inspect diagnostics:    `AspMg_min_dist`, `Mg_coord_count`, `Mg_coord_var`.

---