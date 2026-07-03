#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "CRASAR/CRASAR-U-DROIDs"
REPO_TYPE = "dataset"


def main():
    parser = argparse.ArgumentParser(
        description="Download one specific file from CRASAR-U-DROIDs."
    )

    parser.add_argument(
        "--repo-file",
        required=True,
        help="Path of the file inside the Hugging Face dataset repository.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/crasar"),
        help="Local output directory.",
    )

    args = parser.parse_args()

    local_cache_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=args.repo_file,
    )

    output_path = args.output_dir / args.repo_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(local_cache_path, output_path)

    print("Downloaded CRASAR file")
    print("----------------------")
    print(f"Repo file: {args.repo_file}")
    print(f"Local path: {output_path}")


if __name__ == "__main__":
    main()
