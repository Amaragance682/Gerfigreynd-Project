"""Check that the conda environment and the data are set up correctly.

Run it from the repository folder after `conda activate rei505m-brainage`:

    python scripts/check_setup.py

It prints one OK / WARN / FAIL line per check and exits with a non-zero code if
something the project needs is missing. It does not change anything.

AI-generated from claude code due to me just making this for compatability for reuse
"""

from __future__ import annotations

import importlib
import platform
import sys

failures = 0


def report(status: str, message: str) -> None:
    global failures
    if status == "FAIL":
        failures += 1
    print(f"[{status:<4}] {message}")


def check_packages() -> None:
    print(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    for name in ("numpy", "scipy", "pandas", "sklearn", "matplotlib", "mne", "torch"):
        try:
            module = importlib.import_module(name)
            report("OK", f"import {name} {getattr(module, '__version__', '')}".rstrip())
        except ImportError as error:
            report("FAIL", f"cannot import {name}: {error}")


def check_device() -> None:
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
        major, minor = torch.cuda.get_device_capability()
        detail = f"{torch.cuda.get_device_name()} (compute capability {major}.{minor}, CUDA {torch.version.cuda})"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        detail = "Apple GPU via MPS"
    else:
        device = torch.device("cpu")
        detail = "no GPU found; fine for EDA and the feature baseline, slow for CNN training"
        report("WARN", f"PyTorch will use the CPU: {detail}")

    # One forward and backward pass of a small 1-D convolution on an EEG-shaped batch,
    # because a GPU can be "available" yet lack kernels for this PyTorch build.
    try:
        conv = torch.nn.Conv1d(in_channels=20, out_channels=8, kernel_size=7).to(device)
        batch = torch.randn(4, 20, 512, device=device)  # 4 windows, 20 channels, 8 s at 64 Hz
        conv(batch).square().mean().backward()
        report("OK", f"a training step runs on {device}: {detail}")
    except Exception as error:  # noqa: BLE001 - report whatever the backend raises
        report("FAIL", f"a training step failed on {device}: {error}")


def check_data() -> None:
    import pandas as pd
    import torch
    from pathlib import Path
    DATA_DIR = Path(__file__).resolve().parents[1] / "data" 

    print(f"Data folder: {DATA_DIR}")
    if not DATA_DIR.is_dir():
        report("FAIL", "data folder not found; see 'Get the data' in README.md, or set BRAINAGE_DATA_DIR")
        return

    if (DATA_DIR / "anon_codex_age_sex.csv").is_file():
        report("OK", f"labels: {len(pd.read_csv(DATA_DIR / "anon_codex_age_sex.csv"))} recordings in {DATA_DIR / "anon_codex_age_sex.csv"}")
    else:
        report("FAIL", f"missing {DATA_DIR / "anon_codex_age_sex.csv"}")

    split_ids: set[str] = set()
    for split, path in ((s, DATA_DIR / f"segments_h1k_{s}.pt") for s in ("train", "val", "test")):
        if not path.is_file():
            report("FAIL", f"missing {path.name}")
            continue
        entries = torch.load(path, weights_only=True)  # safe loading: plain lists and tensors only
        split_ids |= {entry[4].removesuffix(".edf") for entry in entries}
        report("OK", f"{split} split: {len(entries)} recordings")

    files = sorted((DATA_DIR / "pre_processed").glob("*.pt"))
    if not files:
        report("FAIL", f"no .pt files in {DATA_DIR / "pre_processed"}")
        return
    missing = split_ids - {f.stem for f in files}
    report("OK" if not missing else "FAIL",
           f"{len(files)} EEG files in {DATA_DIR / "pre_processed"}/; {len(missing)} split recordings have no file")

    eeg = torch.load(files[0], weights_only=True)
    n_channels, min_samples = 20, 64 * 901 # hardcoded samples, sampling rate and target duration
    shape_ok = eeg.ndim == 2 and eeg.shape[0] == n_channels and eeg.shape[1] >= min_samples
    report("OK" if shape_ok else "FAIL",
           f"{files[0].name}: shape {tuple(eeg.shape)}, {eeg.dtype} "
           f"(expected {n_channels} channels x at least {min_samples} samples)")


def main() -> int:
    check_packages()
    if failures == 0:
        check_device()
        check_data()
    print("\nAll checks passed." if failures == 0 else f"\n{failures} check(s) failed; see README.md.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
