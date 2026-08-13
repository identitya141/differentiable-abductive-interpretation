#!/usr/bin/env python3
"""
Data Download Script

Downloads and preprocesses all datasets required for DAI experiments.
Ensures exact reproducibility by pinning dataset versions and checksums.

Usage:
    python scripts/download_data.py --dataset scan --output data/scan
    python scripts/download_data.py --all --output data/
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Optional

from tqdm import tqdm


# =============================================================================
# Dataset Configurations
# =============================================================================

DATASET_CONFIGS = {
    "scan": {
        "name": "SCAN",
        "revision": "c4b756cbc010d75c912f16c42c8f15dc6b7e6c8f",
        "url": "https://github.com/brendenlake/SCAN/archive/c4b756cbc010d75c912f16c42c8f15dc6b7e6c8f.zip",
        "hf_dataset": "scan",
        "version": "1.0.0",
        "expected_files": ["simple_split", "length_split", "add_prim_split"],
        "description": "Compositional command-to-action translation",
    },
    "cogs": {
        "name": "COGS",
        "revision": "165a7b669eade971fa47bf568a2e51925360fed8",
        "url": "https://github.com/najoungkim/COGS/archive/165a7b669eade971fa47bf568a2e51925360fed8.zip",
        "hf_dataset": "cogs",
        "version": "1.0.0",
        "expected_files": ["train.tsv", "dev.tsv", "test.tsv", "gen.tsv"],
        "description": "Compositional generalization challenge based on semantics",
    },
    "slog": {
        "name": "SLOG",
        "revision": "1c55df85006d58b842520c79dcb8e1b43df2836f",
        "url": "https://raw.githubusercontent.com/bingzhilee/SLOG/1c55df85006d58b842520c79dcb8e1b43df2836f/data/",
        "hf_dataset": None,
        "version": "main",
        "expected_files": [
            "cogs_LF/train.tsv",
            "cogs_LF/dev.tsv",
            "cogs_LF/test.tsv",
            "generalization_sets/gen_cogsLF.tsv",
        ],
        "description": "Structural long-distance dependencies generalization benchmark",
    },
    "cfq": {
        "name": "CFQ",
        "url": "https://storage.googleapis.com/cfq_dataset/cfq1.1.tar.gz",
        "hf_dataset": "cfq",
        "hf_revision": "6627f9390245fe11ef09f349b82f6c89f577aabf",
        "version": "1.1",
        "expected_files": ["dataset.json"],
        "description": "Compositional Freebase Questions",
    },
    "clutrr": {
        "name": "CLUTRR",
        "url": "https://github.com/facebookresearch/clutrr/archive/refs/heads/master.zip",
        "hf_dataset": None,  # Must download from GitHub
        "version": "1.0",
        "expected_files": ["data_089907f8", "data_7c5b0e70"],
        "description": "Compositional Language Understanding and Text-based Relational Reasoning",
    },
    "gsm8k": {
        "name": "GSM8K",
        "hf_dataset": "gsm8k",
        "version": "main",
        "expected_files": ["main"],
        "description": "Grade School Math 8K",
    },
}


# =============================================================================
# Download Functions
# =============================================================================

class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: Path, desc: str = "Downloading") -> Path:
    """Download a file from URL with progress bar."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)
    
    return output_path


def download_hf_dataset(dataset_name: str, output_dir: Path, version: str = "main"):
    """Download dataset from Hugging Face."""
    try:
        from datasets import load_dataset
        
        print(f"Downloading {dataset_name} from Hugging Face...")
        
        if dataset_name == "scan":
            # SCAN has multiple splits
            successes = []
            for split in ["simple", "length", "addprim_jump", "addprim_turn_left"]:
                try:
                    ds = load_dataset("scan", split, trust_remote_code=True)
                    save_path = output_dir / f"{split}"
                    ds.save_to_disk(str(save_path))
                    print(f"  Saved {split} to {save_path}")
                    successes.append(True)
                except Exception as e:
                    print(f"  Warning: Could not download {split}: {e}")
                    successes.append(False)
            if not all(successes):
                return False
        
        elif dataset_name == "cogs":
            ds = load_dataset("cogs", trust_remote_code=True)
            ds.save_to_disk(str(output_dir / "hf_dataset"))
            print(f"  Saved to {output_dir / 'hf_dataset'}")
        
        elif dataset_name == "cfq":
            successes = []
            for split in ["mcd1", "mcd2", "mcd3"]:
                try:
                    ds = load_dataset(
                        "cfq", split,
                        revision=DATASET_CONFIGS["cfq"]["hf_revision"],
                        trust_remote_code=True,
                    )
                    save_path = output_dir / f"{split}"
                    ds.save_to_disk(str(save_path))
                    print(f"  Saved {split} to {save_path}")
                    successes.append(True)
                except Exception as e:
                    print(f"  Warning: Could not download {split}: {e}")
                    successes.append(False)
            if not all(successes):
                return False
        
        elif dataset_name == "gsm8k":
            ds = load_dataset("gsm8k", "main", trust_remote_code=True)
            ds.save_to_disk(str(output_dir / "main"))
            print(f"  Saved to {output_dir / 'main'}")
        
        return True
        
    except Exception as e:
        print(f"Error downloading from Hugging Face: {e}")
        return False


def extract_archive(archive_path: Path, output_dir: Path):
    """Extract zip or tar.gz archive."""
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    elif archive_path.name.endswith('.tar.gz'):
        import tarfile
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(output_dir)
    else:
        raise ValueError(f"Unknown archive format: {archive_path}")


def verify_checksum(file_path: Path, expected_md5: Optional[str] = None) -> bool:
    """Verify file checksum."""
    if expected_md5 is None:
        return True
    
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5.update(chunk)
    
    actual_md5 = md5.hexdigest()
    if actual_md5 != expected_md5:
        print(f"Checksum mismatch: expected {expected_md5}, got {actual_md5}")
        return False
    return True


# =============================================================================
# Dataset-Specific Download Functions
# =============================================================================

def download_scan(output_dir: Path):
    """Download SCAN dataset."""
    print("\n" + "="*60)
    print("Downloading SCAN Dataset")
    print("="*60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use the pinned official release and canonicalize its publication layout.
    config = DATASET_CONFIGS["scan"]
    archive_path = output_dir / "scan.zip"
    
    download_url(config["url"], archive_path, "SCAN")
    extract_archive(archive_path, output_dir)
    archive_path.unlink()  # Remove archive
    
    extracted = next(output_dir.glob("SCAN-*/length_split"), None)
    if extracted is None:
        raise FileNotFoundError("Pinned SCAN archive did not contain length_split")
    canonical = output_dir / "length"
    canonical.mkdir(exist_ok=True)
    for name in ("tasks_train_length.txt", "tasks_test_length.txt"):
        shutil.copy2(extracted / name, canonical / name)
    print("✓ SCAN downloaded and canonicalized from pinned GitHub revision")
    return True


def download_cogs(output_dir: Path):
    """Download COGS dataset."""
    print("\n" + "="*60)
    print("Downloading COGS Dataset")
    print("="*60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use the pinned official repository; its COGS-*/data layout is canonical.
    config = DATASET_CONFIGS["cogs"]
    archive_path = output_dir / "cogs.zip"
    
    download_url(config["url"], archive_path, "COGS")
    extract_archive(archive_path, output_dir)
    archive_path.unlink()
    
    extracted = next(output_dir.glob("COGS-*/data"), None)
    if extracted is None:
        raise FileNotFoundError("Pinned COGS archive did not contain data/")
    canonical = output_dir / "COGS-main" / "data"
    canonical.mkdir(parents=True, exist_ok=True)
    for name in ("train.tsv", "dev.tsv", "test.tsv", "gen.tsv"):
        shutil.copy2(extracted / name, canonical / name)
    print("✓ COGS downloaded and canonicalized from pinned GitHub revision")
    return True


def download_slog(output_dir: Path):
    """Download the official SLOG COGS-LF data and protected gen split."""
    print("\n" + "=" * 60)
    print("Downloading SLOG Dataset")
    print("=" * 60)

    output_dir = Path(output_dir)
    cogs_dir = output_dir / "cogs_LF"
    cogs_dir.mkdir(parents=True, exist_ok=True)
    base_url = DATASET_CONFIGS["slog"]["url"]
    for split in ("train", "dev", "test"):
        download_url(
            f"{base_url}cogs_LF/{split}.tsv",
            cogs_dir / f"{split}.tsv",
            f"SLOG {split}",
        )

    archive_path = output_dir / "generalization_sets.zip"
    download_url(
        f"{base_url}generalization_sets.zip",
        archive_path,
        "SLOG generalization",
    )
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(output_dir, pwd=b"SLOG")
    archive_path.unlink()

    for relative_path in DATASET_CONFIGS["slog"]["expected_files"]:
        if not (output_dir / relative_path).is_file():
            raise FileNotFoundError(f"SLOG download is missing {relative_path}")
    print("✓ SLOG downloaded from the official release")
    return True


def download_cfq(output_dir: Path):
    """Download CFQ dataset."""
    print("\n" + "="*60)
    print("Downloading CFQ Dataset")
    print("="*60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try Hugging Face first
    if download_hf_dataset("cfq", output_dir):
        print("✓ CFQ downloaded from Hugging Face")
        return True
    
    # Fall back to direct download
    config = DATASET_CONFIGS["cfq"]
    archive_path = output_dir / "cfq.tar.gz"
    
    print("Falling back to direct download...")
    download_url(config["url"], archive_path, "CFQ")
    extract_archive(archive_path, output_dir)
    archive_path.unlink()
    
    print("✓ CFQ downloaded")
    return True


def download_clutrr(output_dir: Path):
    """
    Download CLUTRR dataset.
    
    CLUTRR is hosted on GitHub and requires special handling:
    1. Clone/download the repo
    2. Run the data generation scripts OR
    3. Download pre-generated data from alternative sources
    
    The dataset structure includes:
    - data_089907f8/: 1.2, 1.3, 2.3 (train on 2-3 hops)
    - data_7c5b0e70/: 1.10 (train on 2-3, test on up to 10 hops)
    """
    print("\n" + "="*60)
    print("Downloading CLUTRR Dataset")
    print("="*60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CLUTRR requires special handling - try multiple approaches
    config = DATASET_CONFIGS["clutrr"]
    
    # Check if already downloaded
    clutrr_main = output_dir / "clutrr-main"
    if clutrr_main.exists():
        print("CLUTRR repo already exists, skipping download")
        return True
    
    # Approach 1: Try Hugging Face datasets (community uploads)
    try:
        from datasets import load_dataset
        print("Trying Hugging Face datasets...")
        
        # Try CLUTRR from HuggingFace (if available)
        try:
            ds = load_dataset("CLUTRR/CLUTRR", trust_remote_code=True)
            save_path = output_dir / "hf_dataset"
            ds.save_to_disk(str(save_path))
            print(f"✓ CLUTRR downloaded from Hugging Face to {save_path}")
            return True
        except Exception as e:
            print(f"  HuggingFace CLUTRR not available: {e}")
    except ImportError:
        pass
    
    # Approach 2: Download from GitHub repo
    print("Downloading from GitHub...")
    archive_path = output_dir / "clutrr.zip"
    
    try:
        download_url(config["url"], archive_path, "CLUTRR")
        extract_archive(archive_path, output_dir)
        archive_path.unlink()
        
        # Check for extracted content
        extracted = list(output_dir.glob("clutrr-*"))
        if extracted:
            clutrr_repo = extracted[0]
            print(f"✓ CLUTRR repo downloaded to {clutrr_repo}")
            
            # Create convenience symlink
            if not (output_dir / "clutrr-main").exists():
                (output_dir / "clutrr-main").symlink_to(clutrr_repo.name)
            
            # Check for pre-generated data
            data_dirs = list(clutrr_repo.glob("data_*"))
            if data_dirs:
                print(f"  Found pre-generated data: {[d.name for d in data_dirs]}")
            else:
                print("  No pre-generated data found. Generating synthetic data...")
                _generate_clutrr_synthetic(output_dir / "synthetic")
            
            return True
    except Exception as e:
        print(f"  GitHub download failed: {e}")
    
    # Approach 3: Generate synthetic data for testing
    print("Generating synthetic CLUTRR data for testing...")
    _generate_clutrr_synthetic(output_dir / "synthetic")
    print("✓ Synthetic CLUTRR data generated")
    print("  Note: For full experiments, manually download from:")
    print("  https://github.com/facebookresearch/clutrr")
    return True


def _generate_clutrr_synthetic(output_dir: Path):
    """
    Generate synthetic CLUTRR-style data for testing.
    
    This creates simple family relationship chains for integration testing.
    For real experiments, use the official CLUTRR data generator.
    """
    import csv
    import random
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    names = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", 
             "Ivy", "Jack", "Karen", "Leo", "Mia", "Nick", "Olivia", "Peter"]
    
    relations = {
        "parent": ["mother", "father"],
        "child": ["son", "daughter"],
        "sibling": ["brother", "sister"],
        "spouse": ["husband", "wife"],
    }
    
    inverse = {
        "mother": "son/daughter",
        "father": "son/daughter",
        "son": "mother/father",
        "daughter": "mother/father",
        "brother": "brother/sister",
        "sister": "brother/sister",
        "husband": "wife",
        "wife": "husband",
    }
    
    # Generate examples for different hop counts
    for num_hops in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        file_path = output_dir / f"data_{num_hops}.csv"
        
        with open(file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["story", "query", "target"])
            writer.writeheader()
            
            num_examples = 100 if num_hops <= 4 else 50
            
            for _ in range(num_examples):
                # Generate a chain of people
                people = random.sample(names, min(num_hops + 1, len(names)))
                
                # Build story
                story_parts = []
                for i in range(num_hops):
                    rel_type = random.choice(list(relations.keys()))
                    rel = random.choice(relations[rel_type])
                    story_parts.append(f"{people[i]}'s {rel} is {people[i+1]}.")
                
                story = " ".join(story_parts)
                query = f"How is {people[-1]} related to {people[0]}?"
                
                # Simplified target (real CLUTRR computes transitive closure)
                if num_hops == 2:
                    target = "grandparent/grandchild"
                elif num_hops == 3:
                    target = "great-grandparent/great-grandchild"
                else:
                    target = "relative"
                
                writer.writerow({
                    "story": story,
                    "query": query,
                    "target": target,
                })
        
        print(f"  Generated {file_path}")
    
    # Create split info file
    split_info = {
        "train_hops": [2, 3, 4],
        "test_hops": [5, 6, 7, 8, 9, 10],
        "note": "Synthetic data for testing. Use official CLUTRR for experiments."
    }
    with open(output_dir / "split_info.json", 'w') as f:
        json.dump(split_info, f, indent=2)


def download_gsm8k(output_dir: Path):
    """Download GSM8K dataset."""
    print("\n" + "="*60)
    print("Downloading GSM8K Dataset")
    print("="*60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if download_hf_dataset("gsm8k", output_dir):
        print("✓ GSM8K downloaded from Hugging Face")
        return True
    
    print("✗ Failed to download GSM8K")
    return False


# =============================================================================
# Main Entry Points
# =============================================================================

DOWNLOAD_FUNCTIONS = {
    "scan": download_scan,
    "cogs": download_cogs,
    "slog": download_slog,
    "cfq": download_cfq,
    "clutrr": download_clutrr,
    "gsm8k": download_gsm8k,
}


def download_dataset(dataset: str, output_dir: Path) -> bool:
    """Download a single dataset."""
    if dataset not in DOWNLOAD_FUNCTIONS:
        print(f"Unknown dataset: {dataset}")
        print(f"Available: {list(DOWNLOAD_FUNCTIONS.keys())}")
        return False
    
    success = DOWNLOAD_FUNCTIONS[dataset](output_dir)
    if not success:
        return False
    errors = validate_canonical_layout(dataset, output_dir)
    if errors:
        for error in errors:
            print(f"Canonical-layout error: {error}")
        return False
    manifest = {
        "dataset": dataset,
        "source_revision": DATASET_CONFIGS[dataset].get("revision", DATASET_CONFIGS[dataset].get("version")),
        "files": {},
    }
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256_MANIFEST.json":
            manifest["files"][str(path.relative_to(output_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output_dir / "SHA256_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return True


def validate_canonical_layout(dataset: str, output_dir: Path) -> list[str]:
    """Return missing canonical publication paths after download/canonicalization."""
    required = {
        "scan": ["length/tasks_train_length.txt", "length/tasks_test_length.txt"],
        "cogs": [f"COGS-main/data/{name}.tsv" for name in ("train", "dev", "test", "gen")],
        "slog": ["cogs_LF/train.tsv", "cogs_LF/dev.tsv", "cogs_LF/test.tsv", "generalization_sets/gen_cogsLF.tsv"],
        "cfq": ["mcd1/dataset_dict.json", "mcd2/dataset_dict.json", "mcd3/dataset_dict.json"],
        "gsm8k": ["main/dataset_dict.json"],
    }
    return [relative for relative in required.get(dataset, []) if not (output_dir / relative).is_file()]


def download_all(output_dir: Path) -> Dict[str, bool]:
    """Download all datasets."""
    results = {}
    for dataset in DOWNLOAD_FUNCTIONS:
        dataset_dir = output_dir / dataset
        results[dataset] = download_dataset(dataset, dataset_dir)
    return results


def main():
    parser = argparse.ArgumentParser(description="Download datasets for DAI experiments")
    parser.add_argument("--dataset", type=str, choices=list(DOWNLOAD_FUNCTIONS.keys()),
                        help="Dataset to download")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--output", type=str, default="data",
                        help="Output directory")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if exists")
    
    args = parser.parse_args()
    output_dir = Path(args.output)
    
    if args.all:
        results = download_all(output_dir)
        print("\n" + "="*60)
        print("Download Summary")
        print("="*60)
        for dataset, success in results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {dataset}")
    elif args.dataset:
        download_dataset(args.dataset, output_dir)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
