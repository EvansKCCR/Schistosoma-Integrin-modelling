# Integrin heterodimer stereochemical/Voronoi QC fix

This package contains the strict VoroMQA parsing revision of `integrin_heterodimer_stereochemical_voronoi_qc.py`.

## What was fixed

1. The pipeline now refuses to treat the `voronota-js-voromqa --output-table-file` result as a per-residue table. That file contains global model-level scores.
2. Local VoroMQA residue scores are accepted only from PDB files containing ATOM/HETATM records with scores in the B-factor field.
3. The script deletes stale per-residue CSVs before each VoroMQA run, preventing older global-table outputs from contaminating new runs.
4. A strict validator rejects suspicious local-score tables with too few residue IDs or wrong chain IDs.
5. `interface_mean_bfactor_or_plddt` is reported but is not used in `heterodimer_context_quality_index`.

## Correct VoroMQA outputs expected

A valid `voromqa_per_residue_all_models.csv` should have hundreds of rows per integrin heterodimer, not one row per model. Expected columns include:

- normalized_chain
- normalized_residue_id
- voromqa_dark_score_mean and/or voromqa_light_score_mean
- model
- is_alpha_beta_interface
- is_user_region

## Recommended command

```bash
python integrin_heterodimer_stereochemical_voronoi_qc.py \
  --input-dir heterodimer_models \
  --output-dir heterodimer_validation_results \
  --alpha-chain A \
  --beta-chain B \
  --region-residues metal_site_residues.txt \
  --nproc 4
```

## Output interpretation

Use the generated `heterodimer_validation_summary.csv` only after confirming:

- `voromqa_all_residue_n` is close to the total number of residues in each heterodimer.
- `voromqa_alpha_chain_n` and `voromqa_beta_chain_n` are non-zero.
- `voromqa_interface_n` is non-zero for models with an alpha-beta interface.
- `interface_mean_bfactor_or_plddt_z` is absent from the final summary or at least not used in the heterodimer-context score.
