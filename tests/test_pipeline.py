import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_repository_structure():
    required_dirs = [
        "01_integrin_detection_classification",
        "02_af3_output_evaluation",
        "03_interface_benchmarking",
        "04_rgd_binding_assessment",
        "05_md_post_analysis_plots",
    ]

    for d in required_dirs:
        assert (ROOT / d).exists(), f"Missing directory: {d}"

def test_readme_exists():
    assert (ROOT / "README.md").exists()

def test_license_exists():
    assert (ROOT / "LICENSE").exists()

def test_rgd_pipeline_help():

    script = (
        ROOT /
        "04_rgd_binding_assessment" /
        "rank_integrin_binding_v3_hotspots.py"
    )

    if script.exists():

        result = subprocess.run(
            ["python", str(script), "--help"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
