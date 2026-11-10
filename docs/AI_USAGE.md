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
