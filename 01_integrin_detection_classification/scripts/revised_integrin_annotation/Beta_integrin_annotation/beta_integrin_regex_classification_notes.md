# Schistosome β-integrin regex motif classifier

This package converts the β-integrin motif presentations in `Motif_beta.xlsx` into a regex-compatible motif library and an updated FASTA-based classifier.

## Key biological aliases

- `beta-Int1` = `Smp_089700.1`
- `ITGB2` = `MS3_00009865.1_mrna`

The script normalizes these aliases internally, but the final classification is based on motif presentation rather than the ID label alone.

## Main change from the previous COBALT panel script

The previous script focused mainly on:
- β-tail NPxY/NPxF motifs,
- S/T proximity around NPxF,
- βI-like MIDAS 5-mer classes,
- human/parasite grouping inferred from IDs.

The updated classifier keeps these features but adds a phylogeny/motif-presentation-aware regex library. It computes collapsed motif scores for:

1. `Schistosome_beta_Int1_ITGB2_like`
2. `Human_beta_integrin_like`
3. `beta_integrin_general`

This allows parasite and human β-integrins to be separated using subtype-specific motif presentations rather than broad β-integrin features alone.

## Parasite-specific classification logic

A high-confidence schistosome β-integrin-like call requires:
- parasite motif score ≥ default threshold,
- at least two required parasite motifs,
- parasite score exceeding the human score by the score-delta threshold.

Important parasite diagnostic motifs include:

- `YPVDLYFLTDLSYTM`
- `GNLDSPEGGMDALLQ`
- `QMGFGAFVDKPVFPF`
- `LFASDGGFHLAGDGR`
- `LLIYKLVITIDDRRE`
- `ENMRWEMAENPIFES`
- `PTTNVLNPTFEENGY`
- parasite-only extracellular/cysteine-rich motifs such as `MATYDADYLEVQVFS`, `QTACSCPSCEK[LM]PMP`, and `NQVCGGPQRG[ST]CQCN`

These motifs reflect the shared motif presentation of the schistosome `beta-Int1`/`ITGB2` group in the supplied motif table.

## Human β-integrin classification logic

A high-confidence human β-integrin-like call requires:
- human motif score ≥ default threshold,
- at least two required human motifs,
- human score exceeding the parasite score by the score-delta threshold.

Important human motif presentations include:

- `[SY]P[IV]D[IL]Y[IY]L[MV]D[FLV]S[ANY]SM`
- `[AGR]N[ILR]D[AST]PEGG[FL]DA[IM][LM]Q`
- `[RT][IL]GFG[AKS][FY]V[DE]K[PTV][SV][LMSV]P[FQY]`
- human HDR/SDL/YDR juxtamembrane variants
- C-terminal `NPLY`/`NPIY`-type tail presentations

These features distinguish human β-integrins from schistosome sequences that show `YFLTD`, `GMDALLQ`, `QMGFGAF`, `IDDRRE`, `ENPIF`, and `LNPTF` presentations.

## Usage

```bash
python beta_integrin_scan_and_classify_regex.py \
  --in beta_integrins.fa \
  --outdir beta_integrin_scan_out \
  --motif-library schisto_beta_integrin_regex_motif_library.tsv \
  --excel \
  --plots
```

The input can be `.fa` or `.fasta`.

The motif library argument is optional because the script contains an embedded copy of the same default motif library. Providing the TSV is still recommended for reproducibility.

```bash
python beta_integrin_scan_and_classify_regex.py \
  --in beta_integrins.fasta \
  --outdir beta_integrin_scan_out
```

## Main outputs

- `beta_integrin_features.csv`  
  Per-sequence summary with scores, tail features, MIDAS 5-mer counts and final class.

- `beta_regex_motif_hits.csv`  
  Per-hit motif table with motif ID, subgroup, source class, regex, matched sequence and position.

- `beta_regex_subgroup_scores.csv`  
  Collapsed motif scores by subgroup.

- `beta_tail_motifs.csv`  
  NPxY/NPxF motif table, including S/T flank counts around NPxF motifs.

- `betaI_MIDAS_5mers.csv`  
  DXSXS, DXSXT and DXSX* 5-mer hits.

- `motif_library_used.tsv`  
  Copy of the library used for the analysis.

## Notes on tail detection

The older script assumed a human-like `HDR` boundary. The new version uses:
1. the last hydrophobic C-terminal transmembrane window,
2. an expanded juxtamembrane fallback allowing `HDR`, `DDR`, `YDR` and `SDL`,
3. a final C-terminal fallback window.

This is important because the schistosome β-Int1/ITGB2-like group contains an `IDDRRE`-type presentation rather than a strictly human-like `HDR` motif.
