#!/usr/bin/env python3
"""Integrated QC and benchmarking pipeline for integrin-RGD peptide complexes.

This orchestrator combines the existing scripts in this directory:

* rank_integrin_binding_v3_hotspots.py
* integrin_heterodimer_qc_pipeline.py
* integrin_heterodimer_stereochemical_voronoi_qc.py

It adds PDB discovery, chain auto-detection, per-chain-group staging, dependency
checks, subprocess execution, and a merged model-level summary.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "SEC", "PYL",
}

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
}

SCRIPT_DIR = Path(__file__).resolve().parent
LIGAND_SCRIPT = SCRIPT_DIR / "rank_integrin_binding_v3_hotspots.py"
PACKING_SCRIPT = SCRIPT_DIR / "integrin_heterodimer_qc_pipeline.py"
STEREO_SCRIPT = SCRIPT_DIR / "integrin_heterodimer_stereochemical_voronoi_qc.py"
DEFAULT_REGION_RESIDUES = SCRIPT_DIR / "metal_site_residues.txt"


@dataclass
class ChainInfo:
    chain_id: str
    residue_count: int = 0
    atom_count: int = 0
    sequence: str = ""
    has_mg: bool = False
    residue_names: List[str] = field(default_factory=list)


@dataclass
class ModelInfo:
    path: Path
    alpha_chain: Optional[str]
    beta_chain: Optional[str]
    rgd_chain: Optional[str]
    mg_chain: Optional[str]
    chains: Dict[str, ChainInfo]
    detection_notes: List[str] = field(default_factory=list)

    @property
    def ligand_group_key(self) -> Tuple[str, str, str]:
        return (self.alpha_chain or "", self.beta_chain or "", self.rgd_chain or "")

    @property
    def heterodimer_group_key(self) -> Tuple[str, str]:
        return (self.alpha_chain or "", self.beta_chain or "")


class ProgressPrinter:
    """Small dependency-free progress reporter for long structure batches."""

    def __init__(self, enabled: bool = True, interval_s: float = 30.0):
        self.enabled = enabled
        self.interval_s = max(float(interval_s), 1.0)
        self._last_tick = 0.0

    def section(self, message: str) -> None:
        if self.enabled:
            print(f"\n== {message} ==")

    def step(self, current: int, total: int, message: str) -> None:
        if not self.enabled:
            return
        total = max(total, 1)
        current = min(max(current, 0), total)
        width = 24
        filled = int(round(width * current / total))
        bar = "#" * filled + "-" * (width - filled)
        print(f"[{bar}] {current}/{total} {message}")

    def tick(self, message: str, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if force or now - self._last_tick >= self.interval_s:
            self._last_tick = now
            print(message, flush=True)


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def command_available(name: str) -> bool:
    return shutil.which(name) is not None


def read_pdb_chains(path: Path) -> Dict[str, ChainInfo]:
    residues: Dict[str, Dict[Tuple[str, str], str]] = {}
    atoms: Dict[str, int] = {}
    has_mg: Dict[str, bool] = {}

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            chain_id = (line[21].strip() or "_")
            resseq = line[22:26].strip()
            icode = line[26].strip()
            resname = line[17:20].strip().upper()
            atom_name = line[12:16].strip().upper()
            element = line[76:78].strip().upper() if len(line) >= 78 else ""

            atoms[chain_id] = atoms.get(chain_id, 0) + 1
            residues.setdefault(chain_id, {})
            if resname in STANDARD_AA:
                residues[chain_id][(resseq, icode)] = resname
            if resname == "MG" or atom_name == "MG" or element == "MG":
                has_mg[chain_id] = True

    out: Dict[str, ChainInfo] = {}
    for chain_id in sorted(set(residues) | set(atoms) | set(has_mg)):
        ordered = [name for _, name in sorted(residues.get(chain_id, {}).items(), key=lambda item: residue_sort_key(item[0]))]
        seq = "".join(THREE_TO_ONE.get(name, "X") for name in ordered)
        out[chain_id] = ChainInfo(
            chain_id=chain_id,
            residue_count=len(ordered),
            atom_count=atoms.get(chain_id, 0),
            sequence=seq,
            has_mg=bool(has_mg.get(chain_id, False)),
            residue_names=ordered,
        )
    return out


def residue_sort_key(residue_key: Tuple[str, str]) -> Tuple[int, str, str]:
    resseq, icode = residue_key
    try:
        n = int(resseq)
    except ValueError:
        n = 0
    return n, icode, resseq


def pick_rgd_chain(chains: Dict[str, ChainInfo], explicit: Optional[str] = None) -> Tuple[Optional[str], List[str]]:
    notes: List[str] = []
    if explicit:
        return explicit, notes

    proteinish = [c for c in chains.values() if c.residue_count > 0]
    rgd_hits = [c for c in proteinish if "RGD" in c.sequence]
    if rgd_hits:
        rgd_hits.sort(key=lambda c: (c.residue_count, c.atom_count, c.chain_id))
        chosen = rgd_hits[0].chain_id
        if len(rgd_hits) > 1:
            notes.append("multiple RGD-containing chains found; selected shortest")
        return chosen, notes

    short_peptides = [c for c in proteinish if c.residue_count <= 40]
    if short_peptides:
        short_peptides.sort(key=lambda c: (c.residue_count, c.atom_count, c.chain_id))
        notes.append("no literal RGD motif found; selected shortest peptide-like chain")
        return short_peptides[0].chain_id, notes

    notes.append("no peptide-like RGD chain detected")
    return None, notes


def pick_receptor_chains(
    chains: Dict[str, ChainInfo],
    rgd_chain: Optional[str],
    alpha_explicit: Optional[str] = None,
    beta_explicit: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], List[str]]:
    notes: List[str] = []
    if alpha_explicit and beta_explicit:
        return alpha_explicit, beta_explicit, notes

    candidates = [c for c in chains.values() if c.chain_id != rgd_chain and c.residue_count >= 40]
    candidates.sort(key=lambda c: (-c.residue_count, c.chain_id))

    alpha = alpha_explicit
    beta = beta_explicit

    if alpha is None and candidates:
        alpha = candidates[0].chain_id
    if beta is None:
        remaining = [c for c in candidates if c.chain_id != alpha]
        if remaining:
            beta = remaining[0].chain_id

    if alpha is None or beta is None:
        notes.append("could not infer two receptor chains")
    elif not alpha_explicit or not beta_explicit:
        notes.append("receptor chains inferred from the two longest protein chains")
    return alpha, beta, notes


def pick_mg_chain(chains: Dict[str, ChainInfo], explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit
    mg = [c.chain_id for c in chains.values() if c.has_mg]
    return mg[0] if mg else None


def discover_models(args: argparse.Namespace) -> List[ModelInfo]:
    paths: List[Path] = []
    if args.input_pdb:
        paths.extend(Path(p).resolve() for p in args.input_pdb)
    if args.input_dir:
        input_dir = Path(args.input_dir).resolve()
        paths.extend(sorted(input_dir.glob(args.pattern)))
    paths = sorted({p for p in paths if p.exists() and p.suffix.lower() == ".pdb"})

    if args.limit and args.limit > 0:
        paths = paths[: args.limit]

    models: List[ModelInfo] = []
    for path in paths:
        chains = read_pdb_chains(path)
        rgd_chain, rgd_notes = pick_rgd_chain(chains, args.rgd_chain)
        alpha_chain, beta_chain, receptor_notes = pick_receptor_chains(
            chains, rgd_chain, args.alpha_chain, args.beta_chain
        )
        mg_chain = pick_mg_chain(chains, args.mg_chain)
        notes = rgd_notes + receptor_notes
        if mg_chain is None:
            notes.append("no Mg chain detected")
        models.append(
            ModelInfo(
                path=path,
                alpha_chain=alpha_chain,
                beta_chain=beta_chain,
                rgd_chain=rgd_chain,
                mg_chain=mg_chain,
                chains=chains,
                detection_notes=notes,
            )
        )
    return models


def write_manifest(models: Sequence[ModelInfo], out_csv: Path) -> None:
    rows = []
    for model in models:
        chain_summary = ";".join(
            f"{cid}:{info.residue_count}res/{info.atom_count}atoms"
            for cid, info in sorted(model.chains.items())
        )
        rows.append(
            {
                "model": model.path.name,
                "path": str(model.path),
                "alpha_chain": model.alpha_chain or "",
                "beta_chain": model.beta_chain or "",
                "rgd_chain": model.rgd_chain or "",
                "mg_chain": model.mg_chain or "",
                "chains": chain_summary,
                "notes": "; ".join(model.detection_notes),
            }
        )
    write_csv_rows(out_csv, rows)


def output_complete(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def stereo_output_reusable(path: Path) -> bool:
    if not output_complete(path):
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return True
    return "Conflicting scattering type symbols" not in text


def write_csv_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def stage_models(models: Sequence[ModelInfo], stage_dir: Path) -> Path:
    stage_dir.mkdir(parents=True, exist_ok=True)
    for model in models:
        dest = stage_dir / model.path.name
        if dest.exists() and dest.stat().st_size == model.path.stat().st_size:
            continue
        shutil.copy2(model.path, dest)
    return stage_dir


def run_subprocess(
    cmd: Sequence[str],
    cwd: Path,
    log_path: Path,
    dry_run: bool,
    progress: Optional[ProgressPrinter] = None,
    label: str = "subprocess",
) -> int:
    cwd.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = " ".join(quote_cmd_part(part) for part in cmd)
    if dry_run:
        log_path.write_text(f"DRY RUN\ncwd={cwd}\n{rendered}\n", encoding="utf-8")
        print(f"[dry-run] {rendered}")
        return 0

    print(f"Running: {rendered}")
    start = time.monotonic()
    last_log_line = ""
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"cwd={cwd}\n{rendered}\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert proc.stdout is not None
        output_queue: "queue.Queue[Optional[str]]" = queue.Queue()

        def _read_output() -> None:
            try:
                for out_line in proc.stdout:
                    output_queue.put(out_line)
            finally:
                output_queue.put(None)

        reader = threading.Thread(target=_read_output, daemon=True)
        reader.start()

        while True:
            try:
                line = output_queue.get(timeout=0.5)
            except queue.Empty:
                line = ""

            if line is None:
                break
            if line:
                log.write(line)
                log.flush()
                stripped = line.strip()
                if stripped:
                    last_log_line = stripped[-180:]

            if progress is not None:
                suffix = f"; last log: {last_log_line}" if last_log_line else ""
                progress.tick(
                    f"  {label} still running after {format_duration(time.monotonic() - start)}{suffix}"
                )

        proc.wait()
        reader.join(timeout=1.0)
    if proc.returncode != 0:
        print(f"  failed with exit code {proc.returncode}; see {log_path}")
    elif progress is not None:
        progress.tick(f"  {label} finished in {format_duration(time.monotonic() - start)}", force=True)
    return proc.returncode


def quote_cmd_part(part: object) -> str:
    text = str(part)
    if re.search(r"\s", text):
        return f'"{text}"'
    return text


def dependency_report() -> Dict[str, object]:
    return {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "modules": {
            "numpy": module_available("numpy"),
            "pandas": module_available("pandas"),
            "Bio": module_available("Bio"),
            "MDAnalysis": module_available("MDAnalysis"),
            "joblib": module_available("joblib"),
            "matplotlib": module_available("matplotlib"),
            "freesasa": module_available("freesasa"),
        },
        "external_tools": {
            "TMalign": any(command_available(x) for x in ("TMalign", "TMalign.exe", "TMscore")),
            "molprobity": any(command_available(x) for x in ("molprobity", "phenix.molprobity")),
            "cablam": any(command_available(x) for x in ("cablam", "phenix.cablam")),
            "VoroMQA": any(command_available(x) for x in ("voronota-js-voromqa", "voronota-voromqa")),
        },
    }


def can_run_ligand(report: Dict[str, object]) -> bool:
    modules = report["modules"]
    return all(bool(modules[m]) for m in ("numpy", "pandas", "MDAnalysis", "joblib", "matplotlib"))


def can_run_heterodimer(report: Dict[str, object]) -> bool:
    modules = report["modules"]
    return all(bool(modules[m]) for m in ("numpy", "pandas", "Bio", "matplotlib"))


def can_run_stereo(report: Dict[str, object]) -> bool:
    modules = report["modules"]
    return all(bool(modules[m]) for m in ("numpy", "pandas", "Bio"))


def run_ligand_groups(
    args: argparse.Namespace,
    models: Sequence[ModelInfo],
    out_dir: Path,
    report: Dict[str, object],
    progress: ProgressPrinter,
) -> List[Path]:
    if args.skip_ligand:
        return []
    if not args.dry_run and not can_run_ligand(report):
        print("Skipping ligand benchmark: missing numpy/pandas/MDAnalysis/joblib/matplotlib in this Python.")
        return []

    csvs: List[Path] = []
    groups: Dict[Tuple[str, str, str], List[ModelInfo]] = {}
    for model in models:
        if all(model.ligand_group_key):
            groups.setdefault(model.ligand_group_key, []).append(model)

    total = len(groups)
    progress.section(f"RGD ligand benchmark groups: {total}")
    for i, ((alpha, beta, rgd), group) in enumerate(sorted(groups.items()), start=1):
        progress.step(i, total, f"ligand group {alpha}/{beta}/RGD {rgd} ({len(group)} model(s))")
        group_dir = out_dir / f"group_{i}_{alpha}{beta}_rgd{rgd}"
        input_dir = stage_models(group, group_dir / "input_pdb")
        run_dir = group_dir / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        output_csv = run_dir / "RGD_binder_ranking.csv"
        contact_dir = run_dir / "contact_maps"
        if args.resume and stereo_output_reusable(output_csv):
            print(f"  resume: using existing {output_csv}")
            csvs.append(output_csv)
            continue
        if args.resume and output_complete(output_csv):
            print(f"  resume: regenerating {output_csv} because it contains old MolProbity element-column errors")
        cmd = [
            sys.executable,
            str(LIGAND_SCRIPT),
            "--input_dir",
            str(input_dir),
            "--rgd_chain",
            rgd,
            "--receptor_chains",
            f"{alpha},{beta}",
            "--mg_chain",
            group[0].mg_chain or beta,
            "--output",
            str(output_csv),
            "--contact_map",
            args.contact_map,
            "--contact_frequency",
            args.contact_frequency,
            "--contact_map_dir",
            str(contact_dir),
            "--freq_threshold",
            str(args.freq_threshold),
            "--n_jobs",
            str(args.n_jobs),
            "--log",
            args.log,
            "--quiet_mda",
            "--fast_contacts",
            "--heavy_only",
        ]
        if args.plots:
            cmd.append("--plots")
        if args.report:
            cmd.append("--report")
        if args.use_freesasa:
            cmd.append("--use_freesasa")
        if args.sanitize_pdb:
            cmd.append("--sanitize_pdb")
        code = run_subprocess(
            cmd,
            run_dir,
            group_dir / "ligand_benchmark.log",
            args.dry_run,
            progress=progress,
            label=f"ligand group {i}/{total}",
        )
        if code == 0:
            csvs.append(output_csv)
    return csvs


def run_packing_groups(
    args: argparse.Namespace,
    models: Sequence[ModelInfo],
    out_dir: Path,
    report: Dict[str, object],
    progress: ProgressPrinter,
) -> List[Path]:
    if args.skip_packing:
        return []
    if not args.dry_run and not can_run_heterodimer(report):
        print("Skipping heterodimer packing QC: missing numpy/pandas/Bio/matplotlib in this Python.")
        return []

    csvs: List[Path] = []
    groups: Dict[Tuple[str, str], List[ModelInfo]] = {}
    for model in models:
        if all(model.heterodimer_group_key):
            groups.setdefault(model.heterodimer_group_key, []).append(model)

    total = len(groups)
    progress.section(f"Heterodimer packing QC groups: {total}")
    for i, ((alpha, beta), group) in enumerate(sorted(groups.items()), start=1):
        progress.step(i, total, f"packing group {alpha}/{beta} ({len(group)} model(s))")
        group_dir = out_dir / f"group_{i}_{alpha}{beta}"
        input_dir = stage_models(group, group_dir / "input_pdb")
        run_dir = group_dir / "run"
        output_dir = run_dir / "packing_qc"
        output_csv = output_dir / "heterodimer_packing_interface_summary.csv"
        if args.resume and output_complete(output_csv):
            print(f"  resume: using existing {output_csv}")
            csvs.append(output_csv)
            continue
        cmd = [
            sys.executable,
            str(PACKING_SCRIPT),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--alpha-chain",
            alpha,
            "--beta-chain",
            beta,
        ]
        if args.skip_plots:
            cmd.append("--skip-plots")
        if args.reference:
            cmd.extend(["--reference", str(Path(args.reference).resolve())])
        if args.templates_dir:
            cmd.extend(["--templates-dir", str(Path(args.templates_dir).resolve())])
        if args.domain_map:
            cmd.extend(["--domain-map", str(Path(args.domain_map).resolve())])
        code = run_subprocess(
            cmd,
            run_dir,
            group_dir / "heterodimer_packing.log",
            args.dry_run,
            progress=progress,
            label=f"packing group {i}/{total}",
        )
        if code == 0:
            csvs.append(output_csv)
    return csvs


def run_stereo_groups(
    args: argparse.Namespace,
    models: Sequence[ModelInfo],
    out_dir: Path,
    report: Dict[str, object],
    progress: ProgressPrinter,
) -> List[Path]:
    if args.skip_stereo:
        return []
    if not args.dry_run and not can_run_stereo(report):
        print("Skipping stereochemical/Voronoi QC: missing numpy/pandas/Bio in this Python.")
        return []

    csvs: List[Path] = []
    groups: Dict[Tuple[str, str], List[ModelInfo]] = {}
    for model in models:
        if all(model.heterodimer_group_key):
            groups.setdefault(model.heterodimer_group_key, []).append(model)

    total = len(groups)
    progress.section(f"Stereochemical/Voronoi QC groups: {total}")
    for i, ((alpha, beta), group) in enumerate(sorted(groups.items()), start=1):
        progress.step(i, total, f"stereo group {alpha}/{beta} ({len(group)} model(s))")
        group_dir = out_dir / f"group_{i}_{alpha}{beta}"
        input_dir = stage_models(group, group_dir / "input_pdb")
        run_dir = group_dir / "run"
        output_dir = run_dir / "stereochemical_voronoi_qc"
        region = Path(args.region_residues).resolve() if args.region_residues else None
        output_csv = output_dir / "heterodimer_validation_summary.csv"
        if args.resume and output_complete(output_csv):
            print(f"  resume: using existing {output_csv}")
            csvs.append(output_csv)
            continue
        cmd = [
            sys.executable,
            str(STEREO_SCRIPT),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--alpha-chain",
            alpha,
            "--beta-chain",
            beta,
            "--contact-cutoff",
            str(args.heterodimer_contact_cutoff),
            "--clash-cutoff",
            str(args.clash_cutoff),
            "--nproc",
            str(args.nproc),
            "--keep-metals",
            "--skip-chain-tools",
        ]
        if region and region.exists():
            cmd.extend(["--region-residues", str(region)])
        if args.keep_hetatm:
            cmd.append("--keep-hetatm")
        if args.keep_hydrogens:
            cmd.append("--keep-hydrogens")
        code = run_subprocess(
            cmd,
            run_dir,
            group_dir / "stereochemical_voronoi.log",
            args.dry_run,
            progress=progress,
            label=f"stereo group {i}/{total}",
        )
        if code == 0:
            csvs.append(output_csv)
    return csvs


def combine_prefixed_csvs(csv_paths: Sequence[Path], prefix: str, model_col_candidates: Sequence[str]) -> Dict[str, Dict[str, str]]:
    combined: Dict[str, Dict[str, str]] = {}
    for path in csv_paths:
        for row in read_csv_rows(path):
            model = ""
            for candidate in model_col_candidates:
                if row.get(candidate):
                    model = Path(row[candidate]).name
                    break
            if not model:
                continue
            out = combined.setdefault(model, {})
            for key, value in row.items():
                if key in model_col_candidates:
                    continue
                out[f"{prefix}_{key}"] = value
    return combined


def merged_summary(
    models: Sequence[ModelInfo],
    ligand_csvs: Sequence[Path],
    packing_csvs: Sequence[Path],
    stereo_csvs: Sequence[Path],
    out_csv: Path,
) -> None:
    ligand = combine_prefixed_csvs(ligand_csvs, "ligand", ["model", "Model"])
    packing = combine_prefixed_csvs(packing_csvs, "packing", ["Model", "model"])
    stereo = combine_prefixed_csvs(stereo_csvs, "stereo", ["model", "Model"])

    rows: List[Dict[str, object]] = []
    for model in models:
        row: Dict[str, object] = {
            "model": model.path.name,
            "path": str(model.path),
            "alpha_chain": model.alpha_chain or "",
            "beta_chain": model.beta_chain or "",
            "rgd_chain": model.rgd_chain or "",
            "mg_chain": model.mg_chain or "",
        }
        row.update(ligand.get(model.path.name, {}))
        row.update(packing.get(model.path.name, {}))
        row.update(stereo.get(model.path.name, {}))
        row["integrated_score"] = integrated_score(row)
        rows.append(row)

    rows.sort(key=lambda r: score_sort_value(r.get("integrated_score")), reverse=True)
    write_csv_rows(out_csv, rows)


def score_sort_value(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def integrated_score(row: Dict[str, object]) -> str:
    components = [
        numeric(row.get("ligand_Rank_score")),
        numeric(row.get("packing_Packing_quality_index")),
        numeric(row.get("stereo_heterodimer_validation_index")),
    ]
    vals = [v for v in components if v is not None]
    if not vals:
        return ""
    return f"{sum(vals) / len(vals):.6g}"


def numeric(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_run_summary(
    out_dir: Path,
    models: Sequence[ModelInfo],
    deps: Dict[str, object],
    ligand_csvs: Sequence[Path],
    packing_csvs: Sequence[Path],
    stereo_csvs: Sequence[Path],
    args: argparse.Namespace,
) -> None:
    payload = {
        "model_count": len(models),
        "output_dir": str(out_dir),
        "dry_run": bool(args.dry_run),
        "dependency_report": deps,
        "outputs": {
            "manifest": str(out_dir / "model_chain_manifest.csv"),
            "combined_summary": str(out_dir / "combined_integrin_rgd_benchmark_summary.csv"),
            "ligand_csvs": [str(p) for p in ligand_csvs],
            "packing_csvs": [str(p) for p in packing_csvs],
            "stereo_csvs": [str(p) for p in stereo_csvs],
        },
    }
    (out_dir / "run_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_model_table(models: Sequence[ModelInfo]) -> None:
    for model in models:
        print(
            f"{model.path.name}: alpha={model.alpha_chain or '-'} "
            f"beta={model.beta_chain or '-'} rgd={model.rgd_chain or '-'} "
            f"mg={model.mg_chain or '-'}"
        )
        if model.detection_notes:
            print(f"  notes: {'; '.join(model.detection_notes)}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Integrated QC and benchmarking for integrin-RGD peptide complex PDBs."
    )
    ap.add_argument("--input-dir", default=None, help="Directory containing PDB files.")
    ap.add_argument("--input-pdb", nargs="*", default=None, help="One or more individual PDB files.")
    ap.add_argument("--pattern", default="*.pdb", help="Glob pattern used with --input-dir.")
    ap.add_argument("--output-dir", default="integrin_rgd_qc_benchmark_results", help="Integrated output directory.")
    ap.add_argument("--limit", type=int, default=0, help="Optional maximum number of PDBs for smoke tests.")

    ap.add_argument("--alpha-chain", default=None, help="Override alpha-integrin chain ID.")
    ap.add_argument("--beta-chain", default=None, help="Override beta-integrin chain ID.")
    ap.add_argument("--rgd-chain", default=None, help="Override RGD peptide chain ID.")
    ap.add_argument("--mg-chain", default=None, help="Override Mg chain ID hint.")

    ap.add_argument("--skip-ligand", action="store_true", help="Skip RGD ligand/interface benchmark.")
    ap.add_argument("--skip-packing", action="store_true", help="Skip heterodimer packing QC.")
    ap.add_argument("--skip-stereo", action="store_true", help="Skip stereochemical/Voronoi QC.")
    ap.add_argument("--dry-run", action="store_true", help="Write manifest and commands without running sub-pipelines.")
    ap.add_argument("--resume", action="store_true", help="Reuse existing non-empty sub-pipeline CSVs instead of rerunning them.")
    ap.add_argument("--list-models", action="store_true", help="Print detected chains and exit.")
    ap.add_argument("--require-all", action="store_true", help="Exit with error if dependencies for requested analyses are missing.")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bars and periodic running-status messages.")
    ap.add_argument("--progress-interval", type=float, default=30.0, help="Seconds between live status messages for long subprocesses.")

    ap.add_argument("--reference", default=None, help="Optional reference heterodimer PDB for packing/reference benchmark.")
    ap.add_argument("--templates-dir", default=None, help="Optional template PDB directory for fold conservation.")
    ap.add_argument("--domain-map", default=None, help="Optional domain CSV: domain,chain,start,end.")
    ap.add_argument("--region-residues", default=str(DEFAULT_REGION_RESIDUES) if DEFAULT_REGION_RESIDUES.exists() else None)

    ap.add_argument("--contact-map", choices=["none", "top", "all"], default="all")
    ap.add_argument("--contact-frequency", choices=["none", "subset", "by_ligand", "grid"], default="grid")
    ap.add_argument("--freq-threshold", type=float, default=0.5)
    ap.add_argument("--n-jobs", type=int, default=1, help="Job count for ligand benchmark.")
    ap.add_argument("--nproc", type=int, default=1, help="Worker count for stereochemical QC.")
    ap.add_argument("--plots", action="store_true", help="Enable plots from the sub-pipelines.")
    ap.add_argument("--skip-plots", action="store_true", help="Disable packing plots.")
    ap.add_argument("--report", action="store_true", help="Write ligand top-model report.")
    ap.add_argument("--use-freesasa", action="store_true", help="Use FreeSASA in ligand benchmark if available.")
    ap.add_argument("--sanitize-pdb", action="store_true", help="Sanitize PDB element columns for ligand benchmark.")
    ap.add_argument("--keep-hetatm", action="store_true", help="Keep HETATM records in stereochemical QC cleaning.")
    ap.add_argument("--keep-hydrogens", action="store_true", help="Keep hydrogens in stereochemical QC cleaning.")
    ap.add_argument("--heterodimer-contact-cutoff", type=float, default=5.0)
    ap.add_argument("--clash-cutoff", type=float, default=2.0)
    ap.add_argument("--log", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_dir and not args.input_pdb:
        raise SystemExit("Provide --input-dir and/or --input-pdb.")

    progress = ProgressPrinter(enabled=not args.no_progress, interval_s=args.progress_interval)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    progress.section("Discovering PDB models")
    models = discover_models(args)
    if not models:
        raise SystemExit("No PDB files found.")
    progress.step(len(models), len(models), f"discovered {len(models)} model(s)")

    write_manifest(models, out_dir / "model_chain_manifest.csv")
    deps = dependency_report()
    (out_dir / "dependency_report.json").write_text(json.dumps(deps, indent=2), encoding="utf-8")

    if args.list_models:
        print_model_table(models)
        print(f"\nManifest written to: {out_dir / 'model_chain_manifest.csv'}")
        return 0

    if args.require_all:
        missing = []
        if not args.skip_ligand and not can_run_ligand(deps):
            missing.append("ligand benchmark dependencies")
        if not args.skip_packing and not can_run_heterodimer(deps):
            missing.append("heterodimer packing dependencies")
        if not args.skip_stereo and not can_run_stereo(deps):
            missing.append("stereochemical/Voronoi dependencies")
        if missing:
            raise SystemExit("Missing required dependencies: " + ", ".join(missing))

    total_requested = sum(not flag for flag in (args.skip_ligand, args.skip_packing, args.skip_stereo))
    progress.section(f"Running requested sub-pipelines: {total_requested}")
    completed_stages = 0

    if args.skip_ligand:
        ligand_csvs: List[Path] = []
    else:
        ligand_csvs = run_ligand_groups(args, models, out_dir / "ligand_binding", deps, progress)
        completed_stages += 1
        progress.step(completed_stages, total_requested, "ligand benchmark stage complete")

    if args.skip_packing:
        packing_csvs: List[Path] = []
    else:
        packing_csvs = run_packing_groups(args, models, out_dir / "heterodimer_packing", deps, progress)
        completed_stages += 1
        progress.step(completed_stages, total_requested, "heterodimer packing stage complete")

    if args.skip_stereo:
        stereo_csvs: List[Path] = []
    else:
        stereo_csvs = run_stereo_groups(args, models, out_dir / "stereochemical_voronoi", deps, progress)
        completed_stages += 1
        progress.step(completed_stages, total_requested, "stereochemical/Voronoi stage complete")

    progress.section("Merging outputs")
    combined_csv = out_dir / "combined_integrin_rgd_benchmark_summary.csv"
    merged_summary(models, ligand_csvs, packing_csvs, stereo_csvs, combined_csv)
    progress.step(len(models), len(models), f"merged {len(models)} model row(s)")

    write_run_summary(out_dir, models, deps, ligand_csvs, packing_csvs, stereo_csvs, args)

    print("\nIntegrated integrin-RGD QC/benchmarking complete.")
    print(f"Manifest:         {out_dir / 'model_chain_manifest.csv'}")
    print(f"Dependency check: {out_dir / 'dependency_report.json'}")
    print(f"Combined summary: {combined_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
