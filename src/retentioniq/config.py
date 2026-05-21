from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"

RANDOM_SEED = 42
OBSERVATION_END = "2025-12-31"


def ensure_project_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, ARTIFACT_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)
