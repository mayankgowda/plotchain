#!/usr/bin/env python3
"""
Generate checksums for reproducibility artifacts
"""

import hashlib
from pathlib import Path

def file_checksum(filepath: Path) -> str:
    """Compute SHA256 checksum of a file"""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def main():
    artifacts = [
        "generate_plotchain_v4.py",
        "run_plotchain_v4_eval.py",
        "concurrent_eval.py",
        "data/plotchain_v4/plotchain_v4.jsonl",
    ]
    
    print("=" * 100)
    print("CHECKSUMS FOR REPRODUCIBILITY ARTIFACTS")
    print("=" * 100)
    print()
    
    for artifact in artifacts:
        path = Path(artifact)
        if path.exists():
            checksum = file_checksum(path)
            size = path.stat().st_size
            print(f"{artifact}:")
            print(f"  SHA256: {checksum}")
            print(f"  Size: {size:,} bytes")
            print()
        else:
            print(f"{artifact}: NOT FOUND")
            print()

if __name__ == "__main__":
    main()

