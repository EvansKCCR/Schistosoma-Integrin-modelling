stage1_integrin_detection/
│
├── input/
│   ├── proteome.fa
│   ├── Pfam-A.hmm
├── intermediate/
│   ├── domtblout.txt
│   ├── integrin_candidates.fa
│
├── results/
│   ├── alpha/
│   ├── beta/
│   ├── merged/
│   ├── Pfam_FASTAs/
├── scripts/
│   ├── run_hmmscan.sh
│   ├── extract_integrins.py
│   ├── integrin_alpha_scan_and_classify.py
│   ├── cobalt_integrin_beta_panels.py
│   └── integrin_alpha_beta_merged_pipeline.py

Step 1 — Pfam scan (NEW entry point)
    hmmscan --cpu 8 \
        --domtblout intermediate/domtblout.txt \
        Pfam-A.hmm input/proteome.fa \
        > intermediate/hmmscan.log
Step 2 — Extract integrins
    python extract_integrins.py \
        --domtbl intermediate/domtblout.txt \
        --fasta input/proteome.fa \
        --out intermediate/integrin_candidates.fa
Step 3 — Run merged pipeline
    python scripts/integrin_alpha_beta_merged_pipeline.py \
        --fasta intermediate/integrin_candidates.fa \
        --outdir output \
        --alpha-script scripts/integrin_alpha_scan_and_classify.py \
        --beta-script scripts/cobalt_integrin_beta_panels.py