"""Download time-series datasets to a local directory.

Supports:
- ETT (ETTh1, ETTh2, ETTm1, ETTm2) — from the ETDataset GitHub repo
- Air Quality (UCI Repository)
- Weather (Max Planck Institute / Jena climate dataset)
"""

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

DATASETS = {
    "ett": {
        "dir": "data/ett",
        "kind": "multi_csv",
        "base_url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small",
        "files": ["ETTh1", "ETTh2", "ETTm1", "ETTm2"],
    },
    "air_quality": {
        "dir": "data/air_quality",
        "kind": "zip",
        "url": "https://archive.ics.uci.edu/static/public/360/air+quality.zip",
        "zip_name": "AirQualityUCI.zip",
    },
    "weather": {
        "dir": "data/weather",
        "kind": "zip",
        "url": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
        "zip_name": "jena_climate_2009_2016.csv.zip",
    },
}


def download_ett(cfg: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in cfg["files"]:
        url = f"{cfg['base_url']}/{name}.csv"
        path = out_dir / f"{name}.csv"
        if path.exists():
            print(f"{name}.csv already exists, skipping.")
            continue
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, path)
        print(f"Saved to {path}")


def download_zip_dataset(cfg: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / cfg["zip_name"]

    if not zip_path.exists():
        print(f"Downloading {cfg['zip_name']}...")
        urllib.request.urlretrieve(cfg["url"], zip_path)
        print(f"Saved to {zip_path}")
    else:
        print(f"{cfg['zip_name']} already exists, skipping download.")

    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    print(f"Extracted to {out_dir}")


def download_dataset(name: str) -> None:
    if name not in DATASETS:
        valid = ", ".join(DATASETS)
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {valid}")

    cfg = DATASETS[name]
    out_dir = Path(cfg["dir"])

    if cfg["kind"] == "multi_csv":
        download_ett(cfg, out_dir)
    elif cfg["kind"] == "zip":
        download_zip_dataset(cfg, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download time-series datasets.")
    parser.add_argument(
        "--dataset",
        "-d",
        choices=list(DATASETS) + ["all"],
        default="ett",
        help="Which dataset to download (default: ett).",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        for name in DATASETS:
            print(f"\n=== {name} ===")
            download_dataset(name)
    else:
        download_dataset(args.dataset)


if __name__ == "__main__":
    main()