# AI use log

The course requires that all use of generative AI is documented: plans, prompts, transcripts and
marked artifacts. The group members are responsible for proper attribution. AI may help with code,
analysis, editing and formatting, but **it must not write any part of the report** (honor code). This
log is the source for the report's "AI statement" section and its appendix on AI use. Will be updated on our first weekly meeting.

## How we mark AI-generated work

- Every file written by an AI tool is marked appropriately as AI assisted or which parts were directly 
  generated, 


## AI transcripts

### 2026-11-02: Óli, Claude Code (Claude Opus 5.5)

**Prompts (paraphrased; verbatim in the transcript):**
1. Set up a shared conda environment for Linux, Windows and macOS; explain the project, the data,
   which improvements over the baseline to aim for, and which loss to use.
2. Drop the presentation-topics file; from now on, propose changes in chat before they go into the repo.
3. Does the honor code require transcripts?
4. Should we keep a shared transcripts folder? Trim the scaffold.
5. Review before pushing; export a transcript.

**What the AI did:**
- Read the assignment PDF and the teacher's starter notebooks. Profiled the data (age distribution,
  splits, zero padding, clipping, amplitude by age) with a script kept outside the repo. Looked up
  the dataset paper (Stevenson et al. 2026) and Iyer et al. (2024).
- Explained the task, how to evaluate improvements (per-child MAE, age-balanced MAE, bootstrap
  intervals), and loss options. It suggested Huber loss on log(age), to be tested against squared loss.
- Wrote first versions of environment.yml, scripts/check_setup.py, .gitignore, .gitattributes, the
  README and the project guide. It also wrote pyproject.toml and src/brainage/, which we later removed.
- Moved the data into data/. Built and tested the environment on Óli's Linux machine (RTX 5080),
  and dry-run checked it for Windows and macOS.
- Reviewed our edits and found a bug in check_setup.py. Wrote a transcript exporter (not in the repo).

**Our changes:** trimmed the README, environment.yml and AI log; removed the package scaffold;
fixed check_setup.py.

**Transcript:** docs/ai_transcripts/2026-11-02_oli_claude-code_setup.md

## 2026-11-08–09 — Feature exploration assistance
**Prompts (paraphrased; verbatim in the transcript):** 
1. Why is scipy.signal.welch not working? 
2. Do a spell check, and check if all variable names are correct.
3. FileNotFoundError, how do I use just the name not .edf with pathlib?
4. What does it mean if stage sleep is N1 but relative band power is delta? Does delta not represent deep sleep?
5. Help with fixing subplot when I have more participants.

**What the AI did:**
-Help diagnosing the SciPy interpreter issue
- Reviewing missing variables and typos, correcting filename extensions and metadata lookups
- Helped with saving figures and adjusting the subplot grid.
- Codex explained relative band power. It identified that the hardcoded N1/N2 intervals require verification against actual sleep annotations.
- Saving figures, and adjusting the subplot layout for two recordings per age group.

**Our changes:**
- Fixed the spelling mistakes, scipy problem, subplot according to AI suggestion.
- Look more into other features and different participants to see better how the PSD is repressenting the data. Tried also different window and step size. 


### 2026-11-12: Óli, Claude Code (Claude Opus 5.5) - Baseline setup 

**Prompts (paraphrased; verbatim in the transcript):**
1. Discussion on how age imbalance affects the model
2. Sliding windows explanations and implementation help in code
3. one channel vs all channels distinction and ridgeCV implementation
4. help with finalizing the baseline model as well as sanity checks
5. interim report alignment with the data and baseline

**What the AI did:**
- Explained the task as regression on age, the inputs (hand-crafted features for the baseline, raw
  EEG for the CNN), and how age imbalance affects a regressor. It showed that the imbalance is much
  smaller on a log(age) scale.
- Co-wrote the baseline code: trimming zero padding away, 30 s windows suggestion, Hjorth and band-power 
  features per channel,  the median over windows per child, ridge regression on log(age) and per-group and
  age-balanced MAE.
- Wrote and tested EDA plotting code (age histograms, amplitude vs. age). Not used in the report, just 
  visual material
- Registered the Jupyter kernel, diagnosed why the editor couldn't list kernels (Code - OSS blocks an
  extension API), and gave the configuration fix.
- The dataset statistics quoted in the interim (age-group counts, split counts, ρ ≈ −0.75) were
  computed by the AI in scratch scripts outside the repo.
- helped with the interim report with suggestions, overview and corrections of facts. Help with condensing everything into a single page.
- oli didnt notice that sigga had pushed changes to github so claude helped merge and explain the pushes

**Our changes:** 
- Wrote docs/Baseline_blueprint.md then wrote and ran notebooks/01_baseline.ipynb


### 2026-11-15: Óli, Claude Code (Claude Opus 5.5) - Convolutional neural network setup and training

**Prompts (paraphrased; verbatim in the transcript):**
1. Continue the CNN plan from the previous session: show step 1, data loading and random crops.
2. Explain how random_starts and the other loader functions work, and why they matter.
3. What is the next step after the loader and is there something that may have been missed?
4. CNN explanation when it comes to processing brain age in children and why it was chosen.
5. Accidentally started another graphically intesive process while running training, did it matter?
6. Summarize everything we did and why in this session.

**What the AI did:**
- Read figure 3 of the assignment. Checked recording lengths, zero padding, per-channel amplitude
  and clipping on the training set with scratch scripts outside the repo, then compared it with current results.
- Wrote the data part of scripts/train_cnn.py: loading a split onto the GPU, the global scale,
  random 16 s crops from the non-padded signal, epoch batches, fixed evaluation windows, and the
  `--check` data checks. Tested it on GPU and CPU.
- Overviewed and corrected each lodader function along with a small worked example for each.
- Helped develop the ConvBlock and BrainAgeCNN classes for the model and its checks (check_model), as well as explained   where it departs from figure 3.
- typo fixes and corrections on code comments
- Ran overfit checks on all 20 children, full runs with and without batch norm, a run with cosine learning-decay
- Made the recomendation to use a fixed amount of epochs instead of just picking the best epoch.
- helped summarize all changes made since the last session for upcoming team meeting.

**Our changes:** 
- coworked the loader, model and checks into scripts/train_cnn.py with claude for the upcoming training notebook
- caught issues early that may have affected the training notebook
- epoch handling pre-training 
