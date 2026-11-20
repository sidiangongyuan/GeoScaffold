#!/usr/bin/env python
"""Create the UrbanIng-V2X split used by the GeoScaffold benchmark.

Training and validation use crossing1 and crossing2; testing uses crossing3, an intersection
unseen in training. The split holds 20 training, 7 validation, and 7 test sequences of 200
frames each (4,000 / 1,400 / 1,400 frames). Each split folder receives relative symbolic links
to the converted sequences, so the dataset folder can be moved as a whole.

    python tools/make_split.py --dataset_dir datasets/UrbanIng-V2X-OpenCOOD-format
"""
import argparse
import os
import sys

SPLIT_NAME = "sis_train_crossing1_crossing2_test_crossing3"
SPLIT = {
    "train": [
        "20241126_0001_crossing2_00", "20241126_0008_crossing1_00", "20241126_0008_crossing1_01",
        "20241126_0010_crossing2_00", "20241126_0019_crossing2_00", "20241126_0022_crossing1_08",
        "20241126_0022_crossing1_09", "20241126_0024_crossing1_08", "20241126_0024_crossing1_09",
        "20241126_0024_crossing1_18", "20241126_0024_crossing1_19", "20241126_0025_crossing1_01",
        "20241127_0000_crossing1_00", "20241127_0003_crossing1_08", "20241127_0003_crossing1_09",
        "20241127_0008_crossing1_00", "20241127_0014_crossing2_00", "20241127_0025_crossing2_00",
        "20241127_0026_crossing2_08", "20241127_0026_crossing2_09",
    ],
    "val": [
        "20241126_0004_crossing2_00", "20241126_0013_crossing2_00", "20241126_0014_crossing1_00",
        "20241126_0017_crossing1_00", "20241126_0018_crossing1_00", "20241126_0025_crossing1_00",
        "20241127_0029_crossing2_00",
    ],
    "test": [
        "20241127_0009_crossing3_00", "20241127_0010_crossing3_08", "20241127_0010_crossing3_09",
        "20241127_0011_crossing3_00", "20241127_0012_crossing3_00", "20241127_0024_crossing3_08",
        "20241127_0024_crossing3_09",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset_dir", required=True,
                        help="converted dataset root (output folder of urbaning_to_opencood_format)")
    root = os.path.abspath(parser.parse_args().dataset_dir)

    missing = [name for names in SPLIT.values() for name in names
               if not os.path.isdir(os.path.join(root, name))]
    if missing:
        sys.exit(f"converted sequences missing in {root}: {missing}")

    for split, names in SPLIT.items():
        target = os.path.join(root, "splits", "sis", SPLIT_NAME, split)
        os.makedirs(target, exist_ok=True)
        for name in names:
            source, link = os.path.join(root, name), os.path.join(target, name)
            if os.path.lexists(link):
                if os.path.realpath(link) != os.path.realpath(source):
                    sys.exit(f"{link} exists and points to another location")
                continue
            os.symlink(os.path.relpath(source, target), link)
        extra = sorted(set(os.listdir(target)) - set(names))
        if extra:
            sys.exit(f"{target} contains entries outside the split: {extra}")
        print(f"{split:5s} {len(names):2d} sequences  {target}")


if __name__ == "__main__":
    main()
