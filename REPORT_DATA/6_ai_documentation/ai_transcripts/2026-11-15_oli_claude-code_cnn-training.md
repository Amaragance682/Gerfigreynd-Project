# CNN training code, overfit check and first full run (15 Nov 2026)

Claude Code (model: Claude Opus 5.5), used by Óli. 15 Nov 2026 20:48 to 15 Nov 2026 21:11 (Iceland time).

Exported from the Claude Code session log. Óli's prompts and Claude's replies are verbatim. Claude's tool calls (commands run, files read, written or edited) are listed in the collapsed sections; their output, Claude's internal reasoning and the editor context sent with each prompt are left out. Attached files are named but not included.

---

## Óli, 2026-11-15 20:48

> okay im coming back from implementing the train-cnn script. here are some rules and useful info from previous sessions:
>
> ## Who and how to work (read first)
> - User: Óli (Ólafur Sær Sigursteinsson, GitHub Amaragance682), use they/them. Teammates: Sigga
>   (Sigríður María Jónsdóttir, macOS, uses Codex) and Amandine (Amandine Sophie S Thouvenot).
> - Óli wants to do the hands-on work. You act as an independent partner:
>   - Before creating or changing ANY file in the repo, show the proposed content in chat and wait
>     for a go-ahead. Óli types or applies it.
>   - Never commit or push; Óli does all git.
>   - Docs are written as text in chat; Óli puts them in docs/ themselves.
>   - Don't finalise docs/PROJECT_GUIDE.md. Ignore the global PRESENTATION_TOPICS.md rule here.
> - Honor code: AI must not write any part of the report. For report text, give ingredients and
>   structure, and review/edit what Óli writes. Example sentences are allowed by Óli's rule, but any
>   that are used must be declared in docs/AI_USAGE.md. Keep track of everything you generate.
> - Test every snippet in your scratchpad on the real data, against Óli's actual file, before
>   showing it. This has caught real problems (a typo, a stuck output buffer, batch-norm noise).
> - Memory files with these rules load automatically, including the transcript-export convention.
>
>
> ## Deadlines and status (today is 15 Nov)
> - Plan targets (docs/PROJECT_PLAN.md): CNN trained 15 Nov (T12). Freeze settings and run the test
>   comparison 16 Nov (T14, T15). Figures and interpretation 18 Nov (T16). Package 19 Nov.
>   Final report (max 10 pages) and code, zipped and emailed to steinng@hi.is, Fri 20 Nov.
> - Done: steps 1–2 (loader and model) in scripts/train_cnn.py, typed in by Óli with their own
>   comments. `python scripts/train_cnn.py --check` passes on GPU and CPU (T11 done).
> - Next: step 3–4 (training code in train_cnn.py), then step 5 (notebooks/02_cnn_results.ipynb).
> - Not committed yet: scripts/train_cnn.py, docs/ai_transcripts/ (3 new files), docs/AI_USAGE.md.
>
> ## Machine and environment
> - Repo /home/amaragance/Desktop/Gerfigreynd-Project (private GitHub, branch main).
> - Python: /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python (torch 2.14.1+cu130).
>   `conda run` swallows output; call the env's python directly, with `-u` when writing to a log.
> - GPU: RTX 5080 Laptop, 16 GB. Óli sometimes plays games that take ~8 GB of it. A game doesn't
>   change results, but can slow runs or crash them out of memory. Ask Óli to close it before real runs.
> - /home is ~94% full; checkpoints are ~2 MB each, which is fine.
>
> ## Data (instructor-preprocessed; unchanged)
> - data/pre_processed/<ID>.pt: float16 (20, 57664) = 20 bipolar channels x 901 s at 64 Hz; one
>   file (26B12Y, train) is 948 s. Cut every file at 901 s. 0–300 s is pre-N2, 300–900 s is N2.
> - Splits: data/segments_h1k_{train,val,test}.pt, 825 / 103 / 104 children.
> - 28 training recordings end in zero padding; real lengths are 587–901 s.
> - Labels: data/anon_codex_age_sex.csv. Ages 0.16–16, binned.
> - Age groups everywhere: <1, 1–3, 3–6, 6–12, 12+ (edges [0,1,3,6,12,inf), left-closed).
>
> ## What scripts/train_cnn.py contains now
> - Constants: FS=64, N_SAMPLES=901*64, WIN=16*64.
> - load_split(split, device): float16 recordings on the GPU, real lengths, ages.
> - global_scale(X, lengths): train RMS over the real signal, 42.41 µV. Divide every split by it.
> - random_starts and crop: random 16 s crops inside the real signal, a whole batch in one GPU op.
> - epoch_batches: each child crops_per_child times, shuffled, in batches.
> - eval_windows(X, lengths): all non-overlapping 16 s windows of the real signal, plus their owners.
> - ConvBlock and BrainAgeCNN (figure 3 of the assignment PDF):
>   - 3 blocks of [Conv2d(1,3)+BN+ReLU] x2 + MaxPool(1,2), widths 16/32/64;
>   - a "fully connected" Conv2d(64, 64, (1,128)) per channel + ReLU, then max over the 20 channels → z (64);
>   - Dropout(0.3) and Linear(64,1) → log(age);
>   - 549k parameters, 95% of them in the fully connected layer. Invariant to channel order.
> - check_data and check_model, run by --check. About 30 ms per step, ~4 s per epoch on the GPU.
>
> ## Decisions for the training code (agreed with Óli, 15 Nov)
> - Fixed 60 epochs, AdamW (lr 1e-3, weight decay 1e-2), CosineAnnealingLR(T_max=60) stepped once
>   per epoch. Use the FINAL model. No early stopping and no best-epoch selection.
>   Reason: val MAE is noisy between epochs. Picking the best epoch on val flatters the CNN, while
>   ridge's alpha was chosen on train (leave-one-out).
> - Recompute batch-norm statistics exactly before every validation and before saving:
>   - reset_running_stats() and momentum=None on every BatchNorm2d;
>   - one no-grad forward pass in train mode over one epoch of training crops, with a FIXED
>     generator so it's deterministic;
>   - then model.eval().
>   Reason: with identical weights, BN's running averages alone moved val MAE between 0.62 and
>   0.69 (balanced 0.82–0.98). With averages over a whole epoch it is stable at 0.67 / 0.97.
> - Always predict in eval mode. In train mode, batch norm on one child's windows normalises the
>   amplitude away, and MAE triples (3.1–3.6).
> - Validation is computed every epoch only for the learning curve, and for choosing settings
>   (window length, loss, lr), never for choosing an epoch.
> - Loss: Huber (delta 1) on log(age). Initialise the head bias to the mean training log(age).
>   8 crops per child per epoch, batch 64, seed 0 (torch.manual_seed and a torch.Generator).
> - Per-child prediction: all eval windows → median of predicted log(age) per child
>   (quantile 0.5) → exp → years.
> - Outputs under outputs/<run_name>/ (git-ignored):
>   - final.pt: state_dict, scale and settings;
>   - learning_curve.csv: epoch, train_loss, val_MAE, val_balanced_MAE, lr, seconds;
>   - val_predictions.csv: id, age, pred.
> - Keep a `--overfit` sanity mode: 20 training children spread over the age range, dropout 0,
>   weight decay 0, 32 crops per child, scored on the same children. T12 asks for "small-run checks".
> - The plan's T12 wording says "best validation checkpoint". It is now the final checkpoint, so
>   tell the team.
>
> ## Draft training code (scratch, not in the repo)
> - /tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/8811b7a3-1944-422a-b36b-7d422eb74514/scratchpad/step2b/training.py
>   has predict_children, mae_scores and train, plus cosine decay. It imports a copy of Óli's file.
> - If /tmp was wiped, the same code is in the "Wrote" sections of
>   docs/ai_transcripts/2026-11-15_oli_claude-code_cnn.md.
> - Still to change:
>   - remove best-epoch saving and patience;
>   - add the BN recompute;
>   - log lr;
>   - write to ROOT/"outputs";
>   - add CLI flags --train, --overfit, --epochs, --run-name.
>   Then show it in chat for Óli to add to scripts/train_cnn.py.
>
> ## Scratch results (AI-run; NOT for the report, Óli reruns from the repo code)
> | validation, MAE in years | MAE | bal. MAE | <1 | 1–3 | 3–6 | 6–12 | 12+ |
> |---|---|---|---|---|---|---|---|
> | ridge, all channels (baseline) | 0.75 | 1.08 | 0.13 | 0.49 | 0.75 | 1.77 | 2.26 |
> | CNN, cosine, best of 60 epochs | 0.62 | 0.82 | 0.16 | 0.44 | 0.76 | 1.33 | 1.42 |
> | same weights, BN averaged over a whole epoch | 0.67 | 0.97 | | | | | |
> | CNN without BN, best epoch | 0.67 | 1.05 | 0.10 | 0.41 | 0.95 | 1.08 | 2.68 |
> - Overfit check: with one fixed crop per child, 20 children are memorised (loss 0.000). With
>   random crops the loss is about 0.07.
> - Unofficial, from earlier: HistGradientBoosting on the baseline features gave 0.57 / 0.74.
>
> ## Next steps
> 1. Show the training additions in chat, tested first in the scratchpad. Óli adds them and runs
>    `--overfit`, then `--train --run-name cnn16s` (~5 min).
> 2. notebooks/02_cnn_results.ipynb:
>    - learning curves (the T12 deliverable);
>    - the evaluate() table with CNN and ridge side by side;
>    - predicted vs. true age, residuals vs. age, MAE per age group;
>    - a paired bootstrap of CNN vs. ridge over the val children.
>    Note: 01_baseline.ipynb does NOT save ridge's per-child predictions. Refit ridge in 02 from
>    outputs/features_win30.csv (seconds), or have 01 save them.
> 3. Settings to try if time allows (val only, ~5 min each): 30 s windows, N2-only crops, MSE on
>    age vs. Huber on log(age) (the plan's loss comparison), no BN. Freeze on 16 Nov.
> 4. T15 test, once: final settings trained on train only (like ridge). Predict test for CNN and
>    ridge with the same table and a paired bootstrap. Optionally 3 seeds to show run-to-run spread.
>
> ## Housekeeping, open
> - Re-export the previous session's transcript (its docs copy stops before the handoff). The
>   exporter is in that session's scratchpad (export_transcript.py); the command is at the end of
>   that transcript. If it's gone, ask Óli where to recreate it.
> - docs/AI_USAGE.md:
>   - paste the CNN entry drafted in the previous session, and extend it with this one;
>   - fix the baseline entry: dates are 11–12 Nov; describe the interim-report review, including
>     two example method sentences the AI wrote at 19:16 on 11 Nov; describe the 12 Nov review
>     of Sigga's push.
> - Add an "AI-assisted" header to scripts/train_cnn.py. Only check_data is marked now.
> - Older items from Sigga's push are in the first message of the previous transcript:
>   data_audit.ipynb bugs, "Segments_" capital S, and the definitions conflict (T07).
>   Her figures were untracked in commit 2543d2f.
>
> so lets start with an overview of what we will be doing and why.

**Claude:**

I'll check the current state first: your file, the draft from last session, and the memory notes.

<details><summary>4 commands, 1 file read</summary>

**Ran:** Check repo status, draft code and memory files

```bash
git status --short && echo --- && wc -l scripts/train_cnn.py && echo --- && ls -la /tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/8811b7a3-1944-422a-b36b-7d422eb74514/scratchpad/step2b/ 2>&1 | head -20 && echo --- && ls /home/amaragance/.claude/projects/-home-amaragance-Desktop-Gerfigreynd-Project/memory/ && ls outputs 2>&1 | head
```

**Read** `scripts/train_cnn.py`

**Ran:** Read the memory notes

```bash
cd /home/amaragance/.claude/projects/-home-amaragance-Desktop-Gerfigreynd-Project/memory/ && for f in *.md; do echo "=== $f"; cat "$f"; done
```

**Ran:** Check git log, the draft training code, and differences from the repo file

```bash
git log --oneline -5 && git status && echo --- && cat /tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/8811b7a3-1944-422a-b36b-7d422eb74514/scratchpad/step2b/training.py && echo --- && diff <(sed 's/[[:space:]]*#.*//' /tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/8811b7a3-1944-422a-b36b-7d422eb74514/scratchpad/step2b/train_cnn.py | grep -v '^\s*$') <(sed 's/[[:space:]]*#.*//' scripts/train_cnn.py | grep -v '^\s*$') | head -40; ls ~/.claude/projects/-home-amaragance-Desktop-Gerfigreynd-Project/*.jsonl | tail -3; ls /tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/*/scratchpad/ 2>/dev/null | head -30
```

**Ran:** Check what the last commit contained and the AI log state

```bash
git show --stat HEAD | head -20; echo ---; grep -n "^#\|CNN\|cnn\|11 Nov\|12 Nov\|2026-11" docs/AI_USAGE.md | head -40; echo; grep -n "T1[2-6]" docs/PROJECT_PLAN.md | head
```

</details>

Everything from last session is now committed (b9aec92) and the working tree is clean, so the "not committed yet" note is out of date. Your `train_cnn.py` matches the copy I tested against, except for `ROOT`. My draft training code and the transcript exporter are still in `/tmp`. A reboot can wipe `/tmp`, so it's best to use them tonight.

## Tonight: finish T12 (due today)

T12 asks for "small-run checks pass; settings, curves and checkpoint are saved". That takes three pieces.

**1. Training code (steps 3–4), added to `scripts/train_cnn.py`**
- `train()`: 60 epochs, AdamW, a learning rate that falls along a cosine curve, Huber loss on log(age), and 8 random crops per child per epoch.
- `predict_children()`: predicts every 16 s window of a child, takes the median, and converts it to years.
- `mae_scores()`: MAE and age-balanced MAE, using the same age groups as the baseline.
- Outputs go to `outputs/<run>/`: `final.pt`, `learning_curve.csv` and `val_predictions.csv`. The new flags are `--overfit`, `--train`, `--epochs` and `--run-name`.

Why the main choices:
- **We keep the final model, not the best epoch.** Val MAE jumps around from epoch to epoch. Picking the best epoch on val would make the CNN look better than it is, and ridge got no such help: its alpha was chosen on train. It also keeps val clean for comparing settings later.
- **We recompute the batch-norm statistics before each validation and before saving.** With the same weights, the batch-norm running averages alone moved val MAE between 0.62 and 0.69. Averaging them over a whole epoch, with a fixed seed, makes the score depend only on the weights.
- **We always predict in eval mode.** In train mode, batch norm on one child's windows removes the amplitude, which is a strong age cue, and MAE triples.

Before showing you anything, I'll finish the remaining edits in my scratchpad and run them against your actual file.

**2. Small-run check: `--overfit`, about 1 minute**
It uses 20 children, no regularisation, and scores them on themselves. If the loss doesn't drop close to zero, something basic is broken: labels, crops, loss or optimiser. This is the "small-run check" T12 asks for.

**3. Full run: `--train --run-name cnn16s`, about 5 minutes**
This produces the learning curve and the checkpoint. Please close any game first. Your run from the repo is the one that counts. My scratch run gave about 0.67 MAE and 0.97 balanced MAE, so expect something close to that.

## 16 Nov: notebook, settings, freeze, test (T14, T15)

**4. `notebooks/02_cnn_results.ipynb`**
- The learning curves (the T12 deliverable).
- A CNN vs. ridge table: MAE, balanced MAE, and MAE per age group.
- Predicted vs. true age, and residuals vs. age.
- A paired bootstrap over the val children. With only 103 children, a gap of 0.08 years could be noise, and the bootstrap shows how sure we can be.
- **Blocker:** `01_baseline.ipynb` doesn't save ridge's per-child predictions. Either we refit ridge in 02 from `outputs/features_win30.csv`, or 01 starts saving them. The baseline is Sigga's (T13/T14), so check with her.

**5. Try 2–3 other settings, on val only, about 5 minutes each**
- Huber on log(age) vs. MSE on age. The plan promises this loss comparison.
- 30 s windows.
- Crops from N2 sleep only.

Then we freeze the settings. Keeping it to a few is partly for time, and partly because every comparison made on val makes val a little less clean.

**6. T15 test, run once**
We train the frozen settings on train only, the same way ridge was trained. Then we predict test for both models and build the same table and bootstrap. If there's time, 3 seeds would show how much the CNN varies from run to run. That tells us whether the CNN vs. ridge gap is real.

After that come figures and interpretation (18 Nov), packaging (19 Nov) and submission (20 Nov).

## Housekeeping (any time before 19 Nov)
- Tell the team that T12's "best validation checkpoint" is now the final checkpoint, and why.
- `docs/AI_USAGE.md`:
  - fix the baseline entry's dates to 11–12 Nov;
  - add the interim-report review, including the two example method sentences I wrote (11 Nov, 19:16);
  - add the 12 Nov review of Sigga's push;
  - extend the CNN entry with tonight's work.
- Add an "AI-assisted" header to `train_cnn.py`. The training code I'm about to draft needs one too.
- Re-export the 15 Nov transcript, since its copy stops before the handoff. The exporter is still in `/tmp`. Export this session at the end as well.
- Older items from Sigga's push: the `data_audit.ipynb` bugs, the capital S in "Segments_", and the definitions conflict (T07).

If that order works for you, I'll start on step 1: finish the training code in my scratchpad, test it against your file, and show it here.

---

## Óli, 2026-11-15 20:58

> okay sounds good, lets check the overfit check on 20 children with no regularization, doing that would be smart before anything else, go ahead with the full run that got interrupted last time because of my game that was open, can you then drop some pseudocode for the cnn results book that i can use if i get lost on trying to implement this? and which functions/classes are relevant from our train-cnn script.
>
> how do i save the ridge per child prediction in 01_baseline to match siggas? and why does that matter?
>
> once we are done with these, then we can move onto 5. and 6.

**Claude:**

I'll start with the GPU state, last session's helper scripts and the baseline notebook.

<details><summary>3 commands</summary>

**Ran:** Check GPU use and last session's helper scripts

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; S=/tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/8811b7a3-1944-422a-b36b-7d422eb74514/scratchpad/step2b; echo --- diag; cat $S/diag.py; echo --- run_cos; cat $S/run_cos.py; echo --- run_full; cat $S/run_full.py; echo --- log tail; tail -5 $S/full_bn_cosine.log; ls -R $S/outputs | head -30
```

**Ran:** List project folders and recent authors

```bash
ls notebooks/ outputs/ docs/ scripts/ data/ | head -60; echo ---; git log --format='%h %an %ad %s' --date=short -12
```

**Ran:** Print the cells of the CNN results and baseline notebooks

```bash
cd notebooks && for nb in 02_cnn_results.ipynb 01_baseline.ipynb; do echo "=================== $nb"; python3 -c "
import json,sys
nb=json.load(open('$nb'))
for i,c in enumerate(nb['cells']):
    src=''.join(c['source'])
    print(f'--- [{i}] {c[\"cell_type\"]}')
    print(src)
"; done
```

</details>

`02_cnn_results.ipynb` couldn't be parsed. I'll check whether it's empty, and look for the prediction format Sigga's work uses.

<details><summary>6 commands</summary>

**Ran:** Check notebook sizes and search the docs for prediction file formats

```bash
cd /home/amaragance/Desktop/Gerfigreynd-Project; ls -la notebooks/; echo ---; grep -rn -i "predict\|\.csv\|to_csv" --include=*.md --include=*.py docs/PROJECT_GUIDE.md docs/PROJECT_PLAN.md docs/Baseline_blueprint.md scripts/feature_exploration.py | grep -v "^docs/ai_transcripts" | head -40
```

**Ran:** Look for prediction or save code in Sigga's notebooks

```bash
cd /home/amaragance/Desktop/Gerfigreynd-Project/notebooks; for nb in feature_extraction.ipynb data_audit.ipynb; do echo "=================== $nb"; python3 -c "
import json
nb=json.load(open('$nb'))
for i,c in enumerate(nb['cells']):
    src=''.join(c['source'])
    if any(k in src for k in ('pred','csv','save','Ridge','ridge','fit(','np.save','to_')):
        print(f'--- [{i}] {c[\"cell_type\"]}'); print(src[:1500])
"; done
```

**Ran:** Search the project docs for agreed prediction formats

```bash
cd /home/amaragance/Desktop/Gerfigreynd-Project; grep -n -i "predict\|csv\|output" docs/PROJECT_GUIDE.md docs/PROJECT_PLAN.md docs/Baseline_blueprint.md | head -50
```

**Ran:** Check last session's overfit results

```bash
S=/tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/8811b7a3-1944-422a-b36b-7d422eb74514/scratchpad/step2b/outputs; head -3 $S/overfit/learning_curve.csv; tail -4 $S/overfit/learning_curve.csv; wc -l $S/overfit/learning_curve.csv; echo; grep -n -i "overfit" /home/amaragance/Desktop/Gerfigreynd-Project/docs/ai_transcripts/2026-11-15_oli_claude-code_cnn.md | head -30
```

**Ran:** Read last session's conclusions from the overfit diagnosis

```bash
sed -n 1433,1560p /home/amaragance/Desktop/Gerfigreynd-Project/docs/ai_transcripts/2026-11-15_oli_claude-code_cnn.md | grep -v "^$" | head -110
```

**Ran:** Build the proposed training code on top of a copy of Óli's script

```bash
mkdir -p /tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/cc8994a0-13be-4156-afde-b0a4b1e5b002/scratchpad/train && cd /tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/cc8994a0-13be-4156-afde-b0a4b1e5b002/scratchpad/train && cat > additions.py <<'EOF'
EDGES = [0, 1, 3, 6, 12, np.inf]   # the age groups of notebooks/01_baseline.ipynb
GROUPS = ["<1", "1-3", "3-6", "6-12", "12+"]


def mae_scores(age, pred):
    # MAE, and age-balanced MAE: the mean of the 5 age groups' MAEs
    err = pd.Series(np.abs(pred - age))
    per_group = err.groupby(pd.cut(age, EDGES, right=False, labels=GROUPS), observed=False).mean()
    return err.mean(), per_group.mean()


def predict_children(model, windows, owner, scale, n_children):
    # every window of every child -> median of the predicted log(age) per child -> years
    model.eval()  # always predict in eval mode; train mode normalises each child's amplitude away
    with torch.no_grad():
        log_pred = torch.cat([model(w.float() / scale) for w in windows.split(512)])
    medians = torch.stack([log_pred[owner == i].quantile(0.5) for i in range(n_children)])
    return medians.exp().cpu().numpy()


def recompute_bn(model, X, lengths, scale, crops_per_child, batch_size, seed):
    # Batch-norm statistics averaged over one whole epoch of training crops, the same crops every time.
    # The usual running averages mostly reflect the last few batches, which moved val MAE by 0.07.
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # a plain average over every batch since the reset
    gen = torch.Generator(device=X.device).manual_seed(seed)
    model.train()
    with torch.no_grad():
        for idx in epoch_batches(len(lengths), crops_per_child, batch_size, gen):
            model(crop(X, idx, random_starts(lengths, idx, gen)).float() / scale)
    model.eval()


def train(device, run_name, epochs=60, lr=1e-3, weight_decay=1e-2, dropout=0.3,
          crops_per_child=8, batch_size=64, seed=0, overfit=False):
    settings = {"epochs": epochs, "lr": lr, "weight_decay": weight_decay, "dropout": dropout,
                "crops_per_child": crops_per_child, "batch_size": batch_size, "seed": seed,
                "overfit": overfit, "window_s": WIN // FS}
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    out = OUTPUTS / run_name
    out.mkdir(parents=True, exist_ok=True)

    ids, X, lengths, ages = load_split("train", device)
    scale = global_scale(X, lengths)
    if overfit:
        # 20 training children spread over the age range, scored on themselves: can the model memorise them?
        keep = torch.argsort(ages)[torch.linspace(0, len(ids) - 1, 20).long()]
        ids, X, lengths, ages = [ids[i] for i in keep.tolist()], X[keep], lengths[keep], ages[keep]
        val_ids, Xv, lv, val_ages = ids, X, lengths, ages
    else:
        val_ids, Xv, lv, val_ages = load_split("val", device)
    windows, owner = eval_windows(Xv, lv)
    val_age = val_ages.cpu().numpy()

    model = BrainAgeCNN(dropout=dropout).to(device)
    with torch.no_grad():
… (71 more lines)
```

</details>

Now the overfit check: 20 children, no dropout, no weight decay, 60 epochs.

<details><summary>1 command</summary>

**Ran:** Run the 20-child overfit check for 60 epochs

```bash
cd /tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/cc8994a0-13be-4156-afde-b0a4b1e5b002/scratchpad/train && /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python -u train_cnn.py --overfit 2>&1 | grep -v Warning | awk 'NR<=3 || NR%6==0 || /final|Error|Traceback/'
```

</details>

The scratch copy's `ROOT` points at my scratchpad because the file lives there. I'll point it back at the repo, in the scratch copy only.

<details><summary>6 commands</summary>

**Ran:** Fix the scratch copy's root and rerun the overfit check

```bash
cd /tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/cc8994a0-13be-4156-afde-b0a4b1e5b002/scratchpad/train && sed -i 's#^ROOT = Path(__file__).resolve().parents\[1\].*#ROOT = Path("/home/amaragance/Desktop/Gerfigreynd-Project")  # SCRATCH ONLY#' train_cnn.py && diff train_cnn_repo_version.py train_cnn.py; /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python -u train_cnn.py --overfit 2>&1 | grep -v Warning | awk 'NR<=3 || NR%6==0 || /final|Error|Traceback/'
```

**Ran:** Point the scratch copy at the repo data and rerun the overfit check

```bash
cd /tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/cc8994a0-13be-4156-afde-b0a4b1e5b002/scratchpad/train && python3 -c "
s = open('train_cnn.py').read()
old = 'ROOT = Path(__file__).resolve().parents[1]'
assert s.count(old) == 1
s = s.replace(old, 'ROOT = Path(\"/home/amaragance/Desktop/Gerfigreynd-Project\")  # SCRATCH ONLY;', 1)
open('train_cnn.py', 'w').write(s)" && diff train_cnn_repo_version.py train_cnn.py; /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python -u train_cnn.py --overfit 2>&1 | grep -v Warning | awk 'NR<=3 || NR%6==0 || /final|Error|Traceback/'
```

**Ran:** Run the overfit check for 150 epochs

```bash
cd /tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/cc8994a0-13be-4156-afde-b0a4b1e5b002/scratchpad/train && /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python -u train_cnn.py --overfit --epochs 150 2>&1 | grep -v Warning | awk 'NR<=2 || NR%15==0 || /final|Error|Traceback/'
```

**Ran:** Show per-child errors of the overfit run

```bash
cd /tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/cc8994a0-13be-4156-afde-b0a4b1e5b002/scratchpad/train && /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python -c "
import pandas as pd, numpy as np
d = pd.read_csv('outputs/overfit/val_predictions.csv')
d['err'] = (d.pred - d.age).round(2); d['rel_%'] = (100*(d.pred/d.age - 1)).round(0)
print(d.round(2).to_string(index=False))
print('median |relative error| %:', np.median(np.abs(d['rel_%'])))
"
```

**Ran:** Run the strict overfit check with one fixed window per child

```bash
cd /tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/cc8994a0-13be-4156-afde-b0a4b1e5b002/scratchpad/train && python3 - <<'EOF'
old = '''        # 20 training children spread over the age range, scored on themselves: can the model memorise them?
        keep = torch.argsort(ages)[torch.linspace(0, len(ids) - 1, 20).long()]
        ids, X, lengths, ages = [ids[i] for i in keep.tolist()], X[keep], lengths[keep], ages[keep]
'''
new = '''        # 20 training children spread over the age range, each cut to one fixed window (the first 16 s of N2).
        # A working pipeline memorises these 20 inputs: the loss goes to about 0 and predictions match the ages.
        keep = torch.argsort(ages)[torch.linspace(0, len(ids) - 1, 20).long()]
        ids, ages = [ids[i] for i in keep.tolist()], ages[keep]
        X, lengths = X[keep, :, 300 * FS:300 * FS + WIN], torch.full_like(lengths[keep], WIN)
'''
for f in ("additions.py", "train_cnn.py", "train_cnn_repo_version.py"):
    s = open(f).read(); assert s.count(old) == 1, f
    open(f, "w").write(s.replace(old, new))
EOF
/home/amaragance/miniforge3/envs/rei505m-brainage/bin/python -u train_cnn.py --overfit 2>&1 | grep -v Warning | awk 'NR<=3 || NR%6==0 || /final|Error|Traceback/'; /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python -c "
import pandas as pd; d = pd.read_csv('outputs/overfit/val_predictions.csv'); print('largest |error|:', (d.pred-d.age).abs().max().round(3), 'years')"
```

**Ran:** Run the full 60-epoch training in the background

```bash
cd /tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/cc8994a0-13be-4156-afde-b0a4b1e5b002/scratchpad/train && nvidia-smi --query-gpu=memory.used --format=csv,noheader && /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python -u train_cnn.py --train --run-name cnn16s > cnn16s.log 2>&1; grep -v Warning cnn16s.log | awk 'NR<=2 || NR%5==0 || /final|Error|Traceback/'
```

</details>

While it trains, I'll test the ridge-saving snippet. It recreates the baseline notebook's cells on the saved feature table and checks that ridge still gives 0.75 / 1.08.

<details><summary>1 command</summary>

**Ran:** Test saving ridge's per-child val predictions from the baseline code

```bash
mkdir -p /tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/cc8994a0-13be-4156-afde-b0a4b1e5b002/scratchpad/nb && cd /tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/cc8994a0-13be-4156-afde-b0a4b1e5b002/scratchpad/nb && cat > ridge_save_test.py <<'EOF'
# Cells 1, 5 and 6 of notebooks/01_baseline.ipynb (features read from the saved CSV), plus the proposed save line.
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

DATA = Path("/home/amaragance/Desktop/Gerfigreynd-Project/data")
OUT = Path(__file__).parent / "outputs"     # repo notebook: Path("../outputs")
features = pd.read_csv("/home/amaragance/Desktop/Gerfigreynd-Project/outputs/features_win30.csv", index_col="id")
splits = {}
for s in ("train", "val", "test"):
    seg = torch.load(DATA / f"segments_h1k_{s}.pt", weights_only=True)
    splits[s] = pd.DataFrame({"id": [t[4].removesuffix(".edf") for t in seg], "age": [t[3] for t in seg]})
EDGES = [0, 1, 3, 6, 12, np.inf]; GROUPS = ["<1", "1-3", "3-6", "6-12", "12+"]
def evaluate(age, pred):
    err = np.abs(pred - age)
    group = pd.cut(age, EDGES, right=False, labels=GROUPS)
    per_group = pd.Series(err).groupby(group, observed=False).mean()
    return {"MAE": err.mean(), "balanced MAE": per_group.mean(), **per_group.to_dict(),
            "r": np.corrcoef(age, pred)[0, 1], "R2": r2_score(age, pred)}
def xy(split, columns):
    df = splits[split]
    return features.loc[df.id, columns].to_numpy(), df.age.to_numpy()
results = {}
def run(name, columns, model, log_target=True):
    X_tr, y_tr = xy("train", columns); X_va, y_va = xy("val", columns)
    model.fit(X_tr, np.log(y_tr) if log_target else y_tr)
    pred = model.predict(X_va)
    if log_target: pred = np.exp(pred)
    results[name] = evaluate(y_va, pred)
    return pred
def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
ALL = list(features.columns)
y_va = splits["val"].age.to_numpy()
pred = run("ridge, all channels", ALL, ridge())

# ---- proposed addition to cell 6 ----
(OUT / "ridge").mkdir(parents=True, exist_ok=True)
pd.DataFrame({"id": splits["val"].id, "age": y_va, "pred": pred}).to_csv(OUT / "ridge" / "val_predictions.csv", index=False)
# -------------------------------------

print(pd.DataFrame(results).T.round(2))
back = pd.read_csv(OUT / "ridge" / "val_predictions.csv")
print(back.head(3)); print(len(back), "rows; ids unique:", back.id.is_unique)
# the saved file alone reproduces the table row
print("from the file:", {k: round(v, 2) for k, v in evaluate(back.age.to_numpy(), back.pred.to_numpy()).items()})
EOF
/home/amaragance/miniforge3/envs/rei505m-brainage/bin/python ridge_save_test.py 2>&1 | grep -v Warning
```

</details>

The ridge snippet works: it reproduces the baseline row (0.75 / 1.08), and the saved file alone gives the same numbers. Next I'll check the val age-group sizes, which matter for the bootstrap design.

<details><summary>2 commands</summary>

**Ran:** Count val children per age group

```bash
cd /tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/cc8994a0-13be-4156-afde-b0a4b1e5b002/scratchpad/nb && /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python -c "
import pandas as pd, numpy as np
d = pd.read_csv('outputs/ridge/val_predictions.csv')
print(pd.cut(d.age, [0,1,3,6,12,np.inf], right=False, labels=['<1','1-3','3-6','6-12','12+']).value_counts(sort=False).to_dict())
"
```

**Ran:** Write a runnable draft of the results notebook logic

```bash
cd /tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/cc8994a0-13be-4156-afde-b0a4b1e5b002/scratchpad/nb && cat > nb02_test.py <<'EOF'
# Runnable version of the 02_cnn_results pseudocode, to check the logic (scratch outputs).
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/cc8994a0-13be-4156-afde-b0a4b1e5b002/scratchpad/train")
from train_cnn import EDGES, GROUPS, BrainAgeCNN, load_split, eval_windows, predict_children

CNN = Path("/tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/cc8994a0-13be-4156-afde-b0a4b1e5b002/scratchpad/train/outputs/cnn16s")
RIDGE = Path(__file__).parent / "outputs" / "ridge"
FIG = Path(__file__).parent / "figs"; FIG.mkdir(exist_ok=True)

def evaluate(age, pred):  # same as 01_baseline (r and R2 left out here)
    err = np.abs(pred - age)
    per_group = pd.Series(err).groupby(pd.cut(age, EDGES, right=False, labels=GROUPS), observed=False).mean()
    return {"MAE": err.mean(), "balanced MAE": per_group.mean(), **per_group.to_dict()}

# 1. learning curve
curve = pd.read_csv(CNN / "learning_curve.csv")
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
a.plot(curve.epoch, curve.train_loss); a.set(xlabel="epoch", ylabel="train loss (Huber, log age)")
b.plot(curve.epoch, curve.val_MAE, label="MAE"); b.plot(curve.epoch, curve.val_balanced_MAE, label="balanced MAE")
b.set(xlabel="epoch", ylabel="val error (years)"); b.legend(); fig.tight_layout(); fig.savefig(FIG / "curve.png")

# 2. one table, both models, joined on id
cnn = pd.read_csv(CNN / "val_predictions.csv")
ridge = pd.read_csv(RIDGE / "val_predictions.csv")
both = cnn.merge(ridge, on="id", suffixes=("_cnn", "_ridge"))
assert len(both) == 103 and np.allclose(both.age_cnn, both.age_ridge)
both = both.rename(columns={"age_cnn": "age"}).drop(columns="age_ridge")
table = pd.DataFrame({"ridge": evaluate(both.age.values, both.pred_ridge.values),
                      "CNN": evaluate(both.age.values, both.pred_cnn.values)}).T.round(2)
print(table)

# 3. stratified paired bootstrap: resample children within each age group, same children for both models
group = pd.cut(both.age, EDGES, right=False, labels=GROUPS)
idx_by_group = [np.flatnonzero(group == g) for g in GROUPS]
err_c, err_r = (both.pred_cnn - both.age).abs().values, (both.pred_ridge - both.age).abs().values
rng = np.random.default_rng(0)
d_mae, d_bal = [], []
for _ in range(10_000):
    pick = [rng.choice(ix, size=len(ix)) for ix in idx_by_group]
    i = np.concatenate(pick)
    d_mae.append(err_c[i].mean() - err_r[i].mean())
    d_bal.append(np.mean([err_c[p].mean() - err_r[p].mean() for p in pick]))
for name, d in [("MAE", np.array(d_mae)), ("balanced MAE", np.array(d_bal))]:
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"CNN - ridge, {name}: {d.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  share of resamples where CNN is worse: {(d > 0).mean():.3f}")

# 4. predicted vs true, residuals vs age
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
for col, label in [("pred_ridge", "ridge"), ("pred_cnn", "CNN")]:
    a.scatter(both.age, both[col], s=12, label=label, alpha=0.7)
    b.scatter(both.age, both[col] - both.age, s=12, label=label, alpha=0.7)
a.plot([0, 17], [0, 17], "k--", lw=1); a.set(xlabel="true age", ylabel="predicted age"); a.legend()
b.axhline(0, color="k", lw=1); b.set(xlabel="true age", ylabel="predicted - true (years)")
fig.tight_layout(); fig.savefig(FIG / "pred.png")

# 5. reload final.pt and check it reproduces the saved predictions (this is what the test step will do)
… (10 more lines)
```

</details>

The full run is training in the background. I'll test the notebook logic on its outputs once it finishes, in about 5 minutes.

<details><summary>1 command</summary>

**Ran:** No-op while the training run continues

```bash
sleep 1; echo waiting-for-notification
```

</details>

The full training run is going in the background, about 5 minutes. When it finishes I'll check the notebook logic on its outputs, then send everything in one reply: the training code to type in, the notebook pseudocode, and the ridge snippet with why it matters.

*(Background task finished: Background command "Run the full 60-epoch training in the background" completed (exit code 0))*

<details><summary>2 commands, 2 files read</summary>

**Ran:** Show the full training run's results

```bash
cat /tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/cc8994a0-13be-4156-afde-b0a4b1e5b002/tasks/bni3683r3.output; cd /tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/cc8994a0-13be-4156-afde-b0a4b1e5b002/scratchpad/train && tail -12 cnn16s.log; ls -la outputs/cnn16s
```

**Ran:** Test the results notebook logic on the scratch outputs

```bash
cd /tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/cc8994a0-13be-4156-afde-b0a4b1e5b002/scratchpad/nb && /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python nb02_test.py 2>&1 | grep -v Warning
```

**Read** `/tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/cc8994a0-13be-4156-afde-b0a4b1e5b002/scratchpad/nb/figs/curve.png`

**Read** `/tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/cc8994a0-13be-4156-afde-b0a4b1e5b002/scratchpad/nb/figs/pred.png`

</details>

Both figures look right, and reloading `final.pt` reproduces the saved predictions exactly. I'll print the exact tested code so what I show you matches what ran.

<details><summary>1 command</summary>

**Ran:** Print the final tested train() header and overfit branch

```bash
cd /tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/cc8994a0-13be-4156-afde-b0a4b1e5b002/scratchpad/train && diff /home/amaragance/Desktop/Gerfigreynd-Project/scripts/train_cnn.py train_cnn_repo_version.py | grep -c '^>' ; sed -n '/^def train(/,/^    windows, owner/p' train_cnn_repo_version.py
```

</details>

I ran both checks in my scratchpad, on a copy of your `train_cnn.py` with the training code added. The numbers are for orientation only: the run that counts is yours from the repo.

## 1. Results

**Overfit check: passes, after one change to how it works.** As we'd agreed it, with random 16 s pieces, the model can't memorise the 20 children. Even after 150 epochs they're still about 10% off, because a wake piece and an N2 piece of the same child look too different. That gives no clear pass or fail.

So I changed it to the standard version: each child gets **one fixed window**, the first 16 s of N2. A working pipeline must memorise those. It did, in about 25 s: loss 0.0006, MAE 0.01 years, and the worst child is off by 0.09 years.

**Full run: 60 epochs, about 5 minutes.** Final val MAE is 0.66 years, balanced 1.03. Over the last 10 epochs it stays between 0.65 and 0.71, so the final model is a sound choice. Reloading `final.pt` reproduces the saved predictions exactly.

| validation (scratch) | MAE | balanced | <1 | 1–3 | 3–6 | 6–12 | 12+ |
|---|---|---|---|---|---|---|---|
| ridge, all channels | 0.75 | 1.08 | 0.13 | 0.49 | 0.75 | 1.77 | 2.26 |
| CNN, final model | 0.66 | 1.03 | 0.12 | 0.34 | 0.94 | 1.13 | 2.63 |

**The paired bootstrap says val can't tell the two models apart yet:**
- CNN minus ridge, MAE: −0.09 years, 95% interval [−0.29, +0.09];
- CNN minus ridge, balanced MAE: −0.05 years, 95% interval [−0.52, +0.38].

That matters for steps 5 and 6: small differences between settings will be noise too.

## 2. Code to add to `scripts/train_cnn.py`

Please mark these parts as AI-assisted in your own way. There are four edits.

**a)** Add numpy before pandas:
```python
import numpy as np
import pandas as pd
```

**b)** Add this under `DATA = ROOT / "data"`:
```python
OUTPUTS = ROOT / "outputs"
```

**c)** Add this between `BrainAgeCNN` and the `# A.I. generated data checker` comment (the empty lines 109–117):
```python
EDGES = [0, 1, 3, 6, 12, np.inf]   # the age groups of notebooks/01_baseline.ipynb
GROUPS = ["<1", "1-3", "3-6", "6-12", "12+"]


def mae_scores(age, pred):
    # MAE, and age-balanced MAE: the mean of the 5 age groups' MAEs
    err = pd.Series(np.abs(pred - age))
    per_group = err.groupby(pd.cut(age, EDGES, right=False, labels=GROUPS), observed=False).mean()
    return err.mean(), per_group.mean()


def predict_children(model, windows, owner, scale, n_children):
    # every window of every child -> median of the predicted log(age) per child -> years
    model.eval()  # always predict in eval mode; train mode normalises each child's amplitude away
    with torch.no_grad():
        log_pred = torch.cat([model(w.float() / scale) for w in windows.split(512)])
    medians = torch.stack([log_pred[owner == i].quantile(0.5) for i in range(n_children)])
    return medians.exp().cpu().numpy()


def recompute_bn(model, X, lengths, scale, crops_per_child, batch_size, seed):
    # Batch-norm statistics averaged over one whole epoch of training crops, the same crops every time.
    # The usual running averages mostly reflect the last few batches, which moved val MAE by 0.07.
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # a plain average over every batch since the reset
    gen = torch.Generator(device=X.device).manual_seed(seed)
    model.train()
    with torch.no_grad():
        for idx in epoch_batches(len(lengths), crops_per_child, batch_size, gen):
            model(crop(X, idx, random_starts(lengths, idx, gen)).float() / scale)
    model.eval()


def train(device, run_name, epochs=60, lr=1e-3, weight_decay=1e-2, dropout=0.3,
          crops_per_child=8, batch_size=64, seed=0, overfit=False):
    settings = {"epochs": epochs, "lr": lr, "weight_decay": weight_decay, "dropout": dropout,
                "crops_per_child": crops_per_child, "batch_size": batch_size, "seed": seed,
                "overfit": overfit, "window_s": WIN // FS}
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    out = OUTPUTS / run_name
    out.mkdir(parents=True, exist_ok=True)

    ids, X, lengths, ages = load_split("train", device)
    scale = global_scale(X, lengths)
    if overfit:
        # 20 training children spread over the age range, each cut to one fixed window (the first 16 s of N2).
        # A working pipeline memorises these 20 inputs: the loss goes to about 0 and predictions match the ages.
        keep = torch.argsort(ages)[torch.linspace(0, len(ids) - 1, 20).long()]
        ids, ages = [ids[i] for i in keep.tolist()], ages[keep]
        X, lengths = X[keep, :, 300 * FS:300 * FS + WIN], torch.full_like(lengths[keep], WIN)
        val_ids, Xv, lv, val_ages = ids, X, lengths, ages
    else:
        val_ids, Xv, lv, val_ages = load_split("val", device)
    windows, owner = eval_windows(Xv, lv)
    val_age = val_ages.cpu().numpy()

    model = BrainAgeCNN(dropout=dropout).to(device)
    with torch.no_grad():
        model.head[-1].bias.fill_(torch.log(ages).mean())  # start by predicting the average log(age)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)  # lr falls smoothly to 0

    log = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for idx in epoch_batches(len(ids), crops_per_child, batch_size, gen):
            x = crop(X, idx, random_starts(lengths, idx, gen)).float() / scale
            loss = nn.functional.huber_loss(model(x), torch.log(ages[idx]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        lr_used = sched.get_last_lr()[0]
        sched.step()

        # val is scored every epoch for the learning curve only, never to pick an epoch
        recompute_bn(model, X, lengths, scale, crops_per_child, batch_size, seed)
        pred = predict_children(model, windows, owner, scale, len(val_ids))
        mae, bal = mae_scores(val_age, pred)
        log.append({"epoch": epoch, "train_loss": np.mean(losses), "val_MAE": mae,
                    "val_balanced_MAE": bal, "lr": lr_used, "seconds": time.time() - t0})
        pd.DataFrame(log).to_csv(out / "learning_curve.csv", index=False)  # every epoch, in case a run dies
        print(f"epoch {epoch:3d}  train loss {np.mean(losses):.4f}  val MAE {mae:.2f}  "
              f"balanced {bal:.2f}  lr {lr_used:.1e}  ({time.time() - t0:.1f} s)")

    # the result is the model after the last epoch, with the BN statistics just recomputed
    torch.save({"model": model.state_dict(), "scale": scale, "settings": settings}, out / "final.pt")
    pd.DataFrame({"id": val_ids, "age": val_age, "pred": pred}).to_csv(out / "val_predictions.csv", index=False)
    print(f"final val MAE {mae:.2f}, balanced {bal:.2f}; saved to {out}")
```

The overfit change works because cutting each recording to exactly one window means `random_starts` always returns 0 and `eval_windows` finds exactly one window. Nothing else needs a special case.

**d)** Replace the `if __name__ == "__main__":` block at the end with:
```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="run the data checks and stop")
    parser.add_argument("--overfit", action="store_true", help="sanity check: memorise 20 training children")
    parser.add_argument("--train", action="store_true", help="train on train, score on val, save to outputs/")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--run-name", default="cnn16s")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda for nvidia gpus, cpu fallback
    if args.check:
        check_data(device)
        check_model(device)
    if args.overfit:
        train(device, "overfit", epochs=args.epochs, dropout=0.0, weight_decay=0.0,
              crops_per_child=32, overfit=True)
    if args.train:
        train(device, args.run_name, epochs=args.epochs)
```

**Then run these from the repo root, with LoreRim closed:**
```
/home/amaragance/miniforge3/envs/rei505m-brainage/bin/python scripts/train_cnn.py --overfit
/home/amaragance/miniforge3/envs/rei505m-brainage/bin/python -u scripts/train_cnn.py --train --run-name cnn16s
```
- The overfit check should end with an MAE of about 0.01–0.02.
- The full run should land very close to 0.66 / 1.03. GPU arithmetic isn't perfectly deterministic, so tiny differences are normal.
- Once both runs work, T12 is done.

## 3. Ridge's per-child predictions in `01_baseline.ipynb`

**About matching Sigga's:** I found no prediction file or format from Sigga in the repo. Her notebooks only compute features, and `01_baseline.ipynb` came in with your commit 2543d2f. So what it should match is the CNN's file: the same columns (`id, age, pred`) and the same layout, `outputs/<model>/val_predictions.csv`. Tell Sigga and Amandine, since T15 and T16 will use this format.

**The edit:** in cell 6, right after `pred = run("ridge, all channels", ALL, ridge())`, add:
```python
(OUT / "ridge").mkdir(parents=True, exist_ok=True)
pd.DataFrame({"id": splits["val"].id, "age": y_va, "pred": pred}).to_csv(OUT / "ridge" / "val_predictions.csv", index=False)
```
- `pred` is in the same order as `splits["val"].id`, because `xy()` picks rows with `features.loc[df.id]`.
- I tested it: the saved file alone reproduces the 0.75 / 1.08 row.
- Cell 4 recomputes all the features. To skip that, use its commented `features = pd.read_csv(...)` line.

**Why it matters:**
1. **A paired comparison.** Both models are scored on the same 103 children, and some children are hard for both. Comparing errors child by child cancels that shared difficulty. Two separate MAE numbers can't, so you couldn't tell whether 0.66 vs. 0.75 is real. As section 1 showed, the bootstrap says it isn't clear yet.
2. **The T16 figures** (predicted vs. true, residuals vs. age, both models in one plot) need a value per child.
3. **T15 requires saved "recording-level test predictions".** The test step reuses this exact format.
4. **One source of truth.** The file holds exactly the predictions behind 01's table. If Sigga changes ridge when she freezes the baseline (T14), rerunning 01 updates the file and 02 picks up the change. Refitting ridge inside 02 would duplicate the code, and the two copies could drift apart.

## 4. Pseudocode for `notebooks/02_cnn_results.ipynb`

**What to import from `train_cnn.py`:**
- `EDGES` and `GROUPS`: the age groups.
- For reloading `final.pt` (cell 7, and the test step later): `BrainAgeCNN`, `load_split`, `eval_windows` and `predict_children`.
- `mae_scores`: only for a quick MAE / balanced MAE.

You don't need `crop`, `random_starts`, `epoch_batches`, `recompute_bn`, `train` or `global_scale`; the scale is stored in `final.pt`. Importing doesn't run any training or load any data, because of the `if __name__` guard.

**Cell 1, setup**
- Add `../scripts` to `sys.path`, then import the names above from `train_cnn`.
- `CNN_RUN = ../outputs/cnn16s` and `RIDGE_RUN = ../outputs/ridge`.
- Copy `evaluate(age, pred)` from cell 5 of 01, so both tables have the same columns (including r and R²).

**Cell 2, learning curve (the T12 deliverable)**
- Read `learning_curve.csv`.
- Left plot: `train_loss` against `epoch`.
- Right plot: `val_MAE` and `val_balanced_MAE` against `epoch`. Optionally show `lr` on a second y-axis.
- Save the figure.

**Cell 3, join the two models**
- Read `cnn = CNN_RUN/val_predictions.csv` and `ridge = RIDGE_RUN/val_predictions.csv`.
- `both = merge on "id"`. Join on id, never on row order.
- Check: 103 rows, and the two age columns are equal.

**Cell 4, table**
- `table = { "ridge": evaluate(age, pred_ridge), "CNN": evaluate(age, pred_cnn) }`, shown as a table with one row per model.

**Cell 5, paired bootstrap, stratified by age group**
```
err_cnn   = |pred_cnn - age|      per child
err_ridge = |pred_ridge - age|    per child
group     = age group of each child
rng = random generator with seed 0
repeat 10,000 times:
    for each age group: draw that group's children with replacement
    (one draw, used for BOTH models; that's what makes it paired)
    d_MAE = mean(err_cnn) - mean(err_ridge) over all drawn children
    d_bal = mean over the 5 groups of (group mean err_cnn - group mean err_ridge)
report for d_MAE and d_bal:
    mean, 95% interval (2.5th and 97.5th percentiles), share of draws where CNN is worse (d > 0)
```
Resample within groups because val has only 7 children aged 12+. A plain bootstrap sometimes draws none of them, and balanced MAE is then undefined.

**Cell 6, figures**
- Predicted vs. true age for both models, with a dashed diagonal.
- Residual (predicted − true) vs. true age, with a line at 0.
- MAE per age group as a bar chart, two bars per group, taken from the table.

**Cell 7, optional check that `final.pt` reproduces the predictions**
```
ck = load final.pt
model = BrainAgeCNN(dropout=ck["settings"]["dropout"]), then load_state_dict(ck["model"]), on the GPU
ids, X, lengths, _ = load_split("val", device)
windows, owner = eval_windows(X, lengths)
pred = predict_children(model, windows, owner, ck["scale"], len(ids))
check: pred matches val_predictions.csv, and ids are in the same order
```
Step 6 uses these same lines, with `"test"` instead of `"val"`.

I ran a working version of all seven cells on the scratch outputs. The figures look right. Both models predict the older children too young, and the CNN does so more for 12+.

## Next
Once your two runs and the ridge line work, we go to steps 5 and 6. Given the bootstrap result, in step 5 we should only change a setting for a clear gain, not for a few hundredths of a year.

For the AI log, I wrote: the training code (including the change to the overfit check), the ridge save lines and the notebook pseudocode. I also ran the overfit check, the full run and a test version of the notebook in my scratchpad.
