"""Run the whole pipeline end to end, in order.

Run from the project root:
    python -m src.run_pipeline                 # steps 1-9, from ingestion
    python -m src.run_pipeline --from evaluate # resume from a later step
    python -m src.run_pipeline --only package  # one step

Each step is its own module and can also be run alone (python -m src.<step>).
A step stops the run if it fails, so later steps never use stale outputs.
"""
import argparse
import subprocess
import sys
import time

from src.config import ROOT

STEPS = [
    ("ingest", "Validate the selected extract and record its hash"),
    ("clean", "Build the adult model-ready file and feature manifest"),
    ("train", "Nested cluster-aware CV model ladder; fit champion"),
    ("evaluate", "Score the locked test set once; fairness, calibration, segments"),
    ("mitigate", "Choose thresholds on training OOF; compare on test"),
    ("robustness", "Threshold stability (bootstrap) and sensitivity sweep"),
    ("explain", "SHAP: global, by gender, and on errors"),
    ("package", "Bundle model + threshold + schema + metadata; reload check"),
    ("schema", "Regenerate the API request schema from the package"),
]
# Module run for each step (default: src.<step>).
MODULES = {"schema": "api.generate_schema"}
NAMES = [name for name, _ in STEPS]


def main():
    parser = argparse.ArgumentParser(description="Run the FinAccess bank-use pipeline.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--from", dest="start", choices=NAMES, help="Start from this step")
    group.add_argument("--only", choices=NAMES, help="Run a single step")
    args = parser.parse_args()

    if args.only:
        selected = [s for s in STEPS if s[0] == args.only]
    else:
        start = NAMES.index(args.start) if args.start else 0
        selected = STEPS[start:]

    started = time.time()
    for i, (name, description) in enumerate(selected, start=1):
        print(f"\n{'=' * 72}\n[{i}/{len(selected)}] {name}: {description}\n{'=' * 72}", flush=True)
        t0 = time.time()
        module = MODULES.get(name, f"src.{name}")
        result = subprocess.run([sys.executable, "-m", module], cwd=ROOT)
        if result.returncode != 0:
            print(f"\nStep '{name}' failed (exit code {result.returncode}). Stopping.")
            sys.exit(result.returncode)
        print(f"-- {name} finished in {time.time() - t0:,.0f}s", flush=True)

    print(f"\nPipeline completed in {(time.time() - started) / 60:,.1f} minutes.")


if __name__ == "__main__":
    main()
