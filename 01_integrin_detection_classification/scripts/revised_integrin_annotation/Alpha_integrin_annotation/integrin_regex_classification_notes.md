# Schistosome α-integrin regex motif classification notes

## Purpose
This update converts the phylogeny-guided motif presentation into a regex-compatible library and integrates that library into the α-integrin motif scanner/classifier.

## Main classification change
The previous feature-derived classifier used broad rules such as HENLA/DIDGDGID, RGD, α-tail motifs, FG-GAP counts, and N-glycosylation density. The updated classifier now uses a weighted regex motif library to resolve schistosome α-integrin subtypes, with special handling for the parasite-specific divergent α2-like and α4-like clades.

## High-confidence subtype logic

### Schistosome α2-like divergent α-integrin
Requires a strong α2-like regex score and at least one required α2-like diagnostic motif. The strongest α2-like features are:

- `SCHISTO_A2_BP1_DIDGDGID`: `F{1,2}GYSIARLGDIDGDGIDDVAISAP`
- `SCHISTO_A2_TM_TAIL_GFFTR`: `VGLWILMIL[SG]ALLYC[CS]GFFTR`

This separates α2-like sequences by the DIDGDGID-containing β-propeller motif and the divergent GFFTR cytoplasmic-tail presentation.

### Schistosome α4-like divergent α-integrin
Requires a strong α4-like regex score and at least one required/near-required α4-like diagnostic motif. The strongest α4-like features are:

- `SCHISTO_A4_BP1_DYDGDNDD`: `[NH]FGF[AS]LTNLGD(?:YDGDG|FDSDG)NDD[IV][AG]VG[AS]P`
- `SCHISTO_A4_BP3_NGNG_GAPSHIFT`: `[NS]GNG[MI]S[VI]LRV[HN][LI]KG[PS][YF][IL]KSV[IV][IM]GAP`
- `SCHISTO_A4_TM_TAIL_GFFRR`: `LGV[IL][CF]LYLLIILLYLLGFFRR`

This separates α4-like sequences by the DYDGDGNDD/FDSDGNDD β-propeller presentation, the shifted NGNG/SGNG-GAP-like block, and the GFFRR tail.

### Schistosome α1-like and α3-like candidates
The classifier resolves the RGD-compatible α1/α3 candidate clades using clade-specific β-propeller motifs and tail variants:

- α1-like core: `RFGHALTNIGD[IMV]DGDGTEDLAVSCP`, `SYFGYSLAVADLDGNGL[AV]DIIVGAP`, `GFFHR`
- α3-like core: `GFGA[AT]ITKLGDINHDG[YF]QD[FL][AI]?[VI]GAP`, `SRFGHSL[LI]F[LI]DINGD[NG]WDDLI[IV]GAP`, `GFFKR`

## New outputs
The updated script writes the original summary files plus:

- `alpha_regex_motif_hits.csv`: every regex-library motif hit per sequence
- `alpha_regex_subgroup_scores.csv`: weighted score table per sequence and subtype
- `alpha_integrin_features.csv`: now includes regex score columns and subtype reasons

## Usage

```bash
python integrin_alpha_scan_and_classify_regex_v2.py \
  --in alpha_integrins.fa \
  --outdir alpha_integrin_scan_out \
  --motif-library schisto_alpha_integrin_regex_motif_library.tsv \
  --excel
```

To export the built-in library from the script:

```bash
python integrin_alpha_scan_and_classify_regex_v2.py \
  --in alpha_integrins.fa \
  --outdir alpha_integrin_scan_out \
  --export-default-motif-library schisto_alpha_integrin_regex_motif_library.tsv
```
