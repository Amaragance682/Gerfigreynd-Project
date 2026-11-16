# Brain age from children's sleep EEG

**Team:** Óli, Sigga, AmaNdine.

| Date | Deliverable |
|---|---|
| Wed 4.11 | Project name emailed to steinng@hi.is |
| Wed 11.11 | Interim report on Gradescope: half a page, no code; EDA and baseline results |
| Fri 20.11 | Final report (PDF, max 10 pages) plus all code, zipped and emailed to steinng@hi.is |

We also meet the instructor at least once a week (Wednesday or Friday slots, or Zoom).

## Set up (once per computer)

1. **Install conda** if you don't have it. [Miniforge](https://github.com/conda-forge/miniforge) 
2. **Create the environment** from the repository folder:
   ```
   git clone <repo-url>
   cd <repo-folder>
   conda env create -f environment.yml
   conda activate rei505m-brainage
   ```
3. **Get the data** from canvas
4. **Check everything:** `python scripts/check_setup.py` should end with `All checks passed.`
5. **Notebooks:** run `jupyter lab`, or in VS Code select the kernel `rei505m-brainage`.

## Get the data

Data is on canvas, move them to this repo in this structure:

```
data/
  anon_codex_age_sex.csv     from the starter pack's data/ folder
  segments_h1k_train.pt      from the starter pack's data/ folder
  segments_h1k_val.pt        from the starter pack's data/ folder
  segments_h1k_test.pt       from the starter pack's data/ folder
  pre_processed/             the 1032 .pt recordings
eeg_starter/                 what the teacher provided as a reference
```

## Repository layout

```
environment.yml     shared conda environment (Linux, Windows, macOS)
scripts/            command-line scripts: check_setup.py now; training and evaluation later
notebooks/          exploration notebooks
docs/               PROJECT_GUIDE.md, AI_USAGE.md
data/               the dataset (git-ignored)
eeg_starter/        the teacher's starter pack (git-ignored)
outputs/            checkpoints, logs, figures (git-ignored, created when needed)
```

## Working conventions

- **Load `.pt` files with `torch.load(path, weights_only=True)`** and convert EEG to float32 before
  computing anything
- **The test split is used once, at the end.** Features, hyperparameters and epochs are chosen on the
  validation split.
- **AI use must be documented** (course honor code): mark AI-generated files if made and log each session  
  in[docs/AI_USAGE.md](docs/AI_USAGE.md). AI must not write any part of the report!!!


### macOS: PyTorch crashes with OpenMP Error #15

In our macOS environment, importing torch crashed because multiple
OpenMP runtimes were loaded. Switching OpenBLAS to its pthreads
build resolved it:

    conda install -n rei505m-brainage -c conda-forge "libopenblas=*=*pthreads*"

Restart the notebook kernel after installation.
