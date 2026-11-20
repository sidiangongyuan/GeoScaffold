#!/usr/bin/env python
"""Build every released model on the CPU and load its checkpoint.

    python tools/check_checkpoints.py [--root checkpoints/urbaning]

Run it from the repository root: the DiscoNet configs load the Early Fusion checkpoint as
their teacher through a path relative to the root. Each folder must hold one config.yaml and
one net_epoch*.pth. The script prints missing or unexpected parameter keys per folder and
exits with status 1 when any folder fails.
"""
import argparse
import glob
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # the check runs on the CPU
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from opencood.hypes_yaml import yaml_utils  # noqa: E402
from opencood.tools import train_utils  # noqa: E402


def check(folder: str) -> str:
    files = glob.glob(os.path.join(folder, "net_epoch*.pth"))
    if len(files) != 1:
        return f"expected one net_epoch*.pth, found {len(files)}"
    model = train_utils.create_model(yaml_utils.load_yaml(None, folder))
    state = torch.load(files[0], map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        return (f"{len(result.missing_keys)} missing keys (first: {result.missing_keys[:3]}), "
                f"{len(result.unexpected_keys)} unexpected keys (first: {result.unexpected_keys[:3]})")
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="checkpoints/urbaning")
    root = parser.parse_args().root
    folders = sorted(os.path.dirname(path)
                     for path in glob.glob(os.path.join(root, "*", "*", "config.yaml")))
    if not folders:
        sys.exit(f"no checkpoint folders under {root}")
    failed = 0
    for folder in folders:
        try:
            problem = check(folder)
        except Exception as error:  # report every folder before exiting
            problem = f"{type(error).__name__}: {error}"
        failed += bool(problem)
        print(f"{'FAIL' if problem else 'ok  '} {folder} {problem}", flush=True)
    print(f"{len(folders) - failed}/{len(folders)} folders load without missing or unexpected keys")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
