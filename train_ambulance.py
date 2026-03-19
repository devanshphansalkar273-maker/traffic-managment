"""
train_ambulance.py
==================
Downloads the Roboflow 'ambulance-qasjo' dataset and fine-tunes YOLOv8n on it.

Usage
-----
    python train_ambulance.py                  # interactive: prompts for API key
    python train_ambulance.py --key RF_API_KEY # non-interactive

The trained weights are saved to:
    ambulance_model.pt   (best weights, copied to project root for easy use)

Requirements
------------
    pip install ultralytics roboflow
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys


def _install(pkg: str) -> None:
    import subprocess
    print(f"Installing {pkg} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


def _ensure_deps() -> None:
    for pkg in ("ultralytics", "roboflow"):
        try:
            __import__(pkg)
        except ImportError:
            _install(pkg)


def download_dataset(api_key: str, dest_dir: str = "ambulance_dataset") -> str:
    """
    Download the Roboflow ambulance dataset.
    Returns the path to the data.yaml file.
    """
    from roboflow import Roboflow  # type: ignore

    print("\n── Downloading Roboflow dataset ──────────────────────────────────")
    rf      = Roboflow(api_key=api_key)
    project = rf.workspace("mrme").project("ambulance-qasjo")
    dataset = project.version(1).download("yolov8", location=dest_dir)

    yaml_path = os.path.join(dest_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        # Roboflow sometimes nests the yaml
        for root, _, files in os.walk(dest_dir):
            for f in files:
                if f == "data.yaml":
                    yaml_path = os.path.join(root, f)
                    break

    print(f"Dataset ready. data.yaml: {yaml_path}")
    return yaml_path


def train(yaml_path: str, epochs: int = 50, imgsz: int = 640) -> str:
    """
    Fine-tune YOLOv8n on the ambulance dataset.
    Returns path to best weights.
    """
    from ultralytics import YOLO  # type: ignore

    print("\n── Training YOLOv8n ──────────────────────────────────────────────")
    print(f"  data   : {yaml_path}")
    print(f"  epochs : {epochs}")
    print(f"  imgsz  : {imgsz}")
    print("  This takes ~5–15 min on CPU, ~2–5 min on GPU.\n")

    model   = YOLO("yolov8n.pt")   # start from COCO pretrained weights
    results = model.train(
        data    = yaml_path,
        epochs  = epochs,
        imgsz   = imgsz,
        batch   = 16,
        name    = "ambulance_run",
        exist_ok= True,
        patience= 15,          # early stopping
        plots   = True,
    )

    # Locate best.pt
    best_pt = os.path.join("runs", "detect", "ambulance_run", "weights", "best.pt")
    if not os.path.exists(best_pt):
        # Fallback search
        for root, _, files in os.walk(os.path.join("runs", "detect")):
            if "best.pt" in files:
                best_pt = os.path.join(root, "best.pt")
                break

    dest = "ambulance_model.pt"
    shutil.copy(best_pt, dest)
    print(f"\n── Training complete ─────────────────────────────────────────────")
    print(f"  Best weights saved to: {dest}")
    print(f"  You can now run main.py — it will use this model automatically.")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 ambulance detector")
    parser.add_argument("--key",    default="",  help="Roboflow API key")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz",  type=int, default=640)
    parser.add_argument("--data",   default="",
                        help="Path to existing data.yaml (skip download)")
    args = parser.parse_args()

    _ensure_deps()

    # ── Resolve data.yaml ─────────────────────────────────────────────────────
    if args.data and os.path.exists(args.data):
        yaml_path = args.data
        print(f"Using existing dataset: {yaml_path}")
    else:
        api_key = args.key or os.environ.get("ROBOFLOW_API_KEY", "")
        if not api_key:
            print("─" * 60)
            print("  Roboflow API key required to download the dataset.")
            print("  Get yours free at: https://app.roboflow.com  → Settings → API")
            print("─" * 60)
            api_key = input("  Enter your Roboflow API key: ").strip()
        if not api_key:
            print("No API key provided. Exiting.")
            sys.exit(1)
        yaml_path = download_dataset(api_key)

    # ── Train ─────────────────────────────────────────────────────────────────
    train(yaml_path, epochs=args.epochs, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
