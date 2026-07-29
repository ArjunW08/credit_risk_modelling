"""
Training pipeline entrypoint for Docker.

Runs the full pipeline (ingestion → processing → training) from the
project root directory so that the existing scripts' relative paths
(../../data, ../../logs, etc.) resolve correctly inside the container.
"""

import os
import subprocess
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PIPELINE_STEPS = [
    {
        "name": "Data ingestion",
        "script": os.path.join("scripts", "data", "ingestion_db.py"),
        "cwd": os.path.join(PROJECT_ROOT, "scripts", "data"),
    },
    {
        "name": "Data processing",
        "script": os.path.join("scripts", "data", "data_processing.py"),
        "cwd": os.path.join(PROJECT_ROOT, "scripts", "data"),
    },
    {
        "name": "Model training",
        "script": os.path.join("scripts", "model_creation", "train.py"),
        "cwd": os.path.join(PROJECT_ROOT, "scripts", "model_creation"),
    },
]


def run_step(step: dict) -> None:
    """Execute a single pipeline step as a subprocess."""
    script_path = os.path.join(PROJECT_ROOT, step["script"])
    print(f"\n{'='*60}")
    print(f"  Step: {step['name']}")
    print(f"  Script: {step['script']}")
    print(f"  Working directory: {step['cwd']}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=step["cwd"],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )

    if result.returncode != 0:
        print(f"\n[ERROR] Step '{step['name']}' failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n[OK] Step '{step['name']}' completed successfully.")


def main() -> None:
    print("Credit Risk Modelling — Training Pipeline")
    print(f"Project root: {PROJECT_ROOT}\n")

    for step in PIPELINE_STEPS:
        run_step(step)

    print("\n" + "=" * 60)
    print("  All pipeline steps completed successfully.")
    print("  Model saved to: models/predict_loan_possibility_model.pkl")
    print("=" * 60)


if __name__ == "__main__":
    main()
