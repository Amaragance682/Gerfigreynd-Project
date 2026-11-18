# Frozen settings, test-set comparison, results notebook and the experiment plan (16 Nov 2026)

Claude Code (model: Claude Opus 5.5), used by Óli. 16 Nov 2026 18:41 to 16 Nov 2026 21:30 (Iceland time).

Exported from the Claude Code session log. Óli's prompts and Claude's replies are verbatim. Claude's tool calls (commands run, files read, written or edited) are listed in the collapsed sections; their output, Claude's internal reasoning and the editor context sent with each prompt are left out. Attached files are named but not included.

---

## Óli, 2026-11-16 18:41

> # Handoff: REI505M brain-age project, results and report material (from session of 15–16 Nov 2026)
>
> ## Who and how to work (read first)
> - User: Óli (Ólafur Sær Sigursteinsson, GitHub Amaragance682), use they/them. Teammates: Sigga
>   (Sigríður María Jónsdóttir, macOS, uses Codex) and Amandine (Amandine Sophie S Thouvenot).
> - Óli does the hands-on work; you are an independent partner:
>   - Before creating or changing ANY file in the repo, show the proposed content in chat and wait for
>     a go-ahead. Óli types or applies it. Exception: transcripts Óli asks you to export.
>   - Never commit or push; Óli does all git.
>   - Docs are written as text in chat; Óli puts them in docs/.
>   - Don't finalise docs/PROJECT_GUIDE.md. Ignore the global PRESENTATION_TOPICS.md rule here.
> - Honor code: AI must not write any part of the report, and figure and table captions count as report text.
>   - For each section, give ingredients: points to make, numbers, which table or figure, citations
>     to look up. Then review and edit what Óli writes.
>   - Example sentences are allowed by Óli's rule, but any that are used must be declared in
>     docs/AI_USAGE.md.
>   - Code that produces tables and figures is fine to help with, marked as AI-assisted.
> - Test every snippet in your scratchpad on the real data, against Óli's actual files, before showing it.
> - Memory files load automatically, including the transcript-export convention.
>
> ## Deadlines (today is 16 Nov)
> - Plan (docs/PROJECT_PLAN.md):
>   - T14 freeze settings and T15 test comparison: 16 Nov (not done yet).
>   - T16 figures and interpretation: 18 Nov.
>   - T18 package: 19 Nov.
> - Final report (PDF, max 10 pages excluding appendices, past tense, no code in the text) plus all
>   code, zipped and emailed to steinng@hi.is by Fri 20 Nov.
> - Report sections required by the assignment (REI505M_Final_project.pdf, section 2):
>   - Introduction (0.5–1 page: what, why, how) and prior work (with citations);
>   - AI statement; dataset and processing;
>   - methods (algorithms, why they were chosen, how they were evaluated, how models and
>     hyperparameters were selected);
>   - results (tables and figures with captions, referred to in the text; effects of hyperparameter
>     choice; training/validation curves; experiments that did not work; text in figures not too small);
>   - conclusions and future work; references; collaboration (who did what);
>   - appendix documenting AI use.
>
> ## Machine and environment
> - Repo /home/amaragance/Desktop/Gerfigreynd-Project (private GitHub, branch main). Editor: Code - OSS.
> - Python: /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python (torch 2.14.1+cu130).
>   - After a reboot, Óli's fish terminal needs `conda activate rei505m-brainage`; plain `python` has
>     no pandas.
>   - Use `-u` when writing to a log.
> - Notebooks: use plain `from tqdm import tqdm`. `tqdm.auto` draws a widget, and Code - OSS then stops
>   sending later cells; the cell looks "running" forever. Headless fallback:
>   `jupyter nbconvert --to notebook --execute --inplace notebooks/<nb>.ipynb`.
> - GPU: RTX 5080 Laptop, 16 GB. Óli sometimes runs games that take ~7 GB; it doesn't change results
>   but slows runs. Ask Óli to close them before real runs.
> - /tmp (your scratchpad) is wiped on every reboot; it happened twice on 16 Nov. Nothing durable
>   goes there.
>
> ## Data (instructor-preprocessed; unchanged)
> - data/pre_processed/<ID>.pt: float16 (20, 57664) = 20 bipolar EEG channels x 901 s at 64 Hz.
>   - Band-passed 0.5–30 Hz; cut every file at 901 s (one is 948 s).
>   - 0–300 s is pre-N2, 300–900 s is N2.
>   - No ECG channel in these files.
> - Splits: data/segments_h1k_{train,val,test}.pt, 825 / 103 / 104 children, made by the
>   instructor. No id appears in two splits; the label file has no subject id, so repeat recordings
>   of one child can't be checked.
> - Labels: data/anon_codex_age_sex.csv. Ages 0.16–16, rounded (spacing ~16 days for infants to
>   ~150 days for teens).
>   - Median age: train 2.98, val 2.07, test 2.63.
>   - Under 6: train 567, val 80, test 74.
>   - 15 or over: 9 / 1 / 1.
> - Age groups everywhere: <1, 1–3, 3–6, 6–12, 12+ (left-closed). Val counts 30/32/18/16/7.
>
> ## Code and outputs now (all run by Óli from the repo)
> - notebooks/01_baseline.ipynb:
>   - 13 features per channel (log variance, Hjorth mobility and complexity, log and relative power
>     in delta, theta, alpha, sigma, beta) from 30 s windows;
>   - median over windows, separately for pre-N2 and N2: 520 features;
>   - StandardScaler + RidgeCV on log(age), alpha by leave-one-out on train (picks 56);
>   - writes outputs/features_win30.csv and outputs/ridge/val_predictions.csv (id, age, pred).
>   - Validation table:
>
>     | model | MAE | balanced MAE |
>     |---|---|---|
>     | predict the training median | 2.98 | 4.37 |
>     | ridge, C3-P3 only | 1.03 | 1.34 |
>     | ridge, all channels (THE baseline) | 0.75 | 1.08 |
>     | ridge, pre-N2 only | 1.24 | 1.94 |
>     | ridge, N2 only | 0.78 | 1.29 |

**Claude:**

I'll start by getting oriented: the memory files, the plan, and what's in the repo now, since the handoff stops after the baseline table.

<details><summary>1 command</summary>

**Ran:** Read memory files and list repo state

```bash
M=/home/amaragance/.claude/projects/-home-amaragance-Desktop-Gerfigreynd-Project/memory; for f in $M/*.md; do echo "=== $f"; cat "$f"; done; echo; git status --short; git log --oneline -8; ls docs notebooks outputs outputs/* scripts src 2>/dev/null | head -80
```

</details>

The handoff cuts off after the baseline table, so I'll get the rest from the repo and the latest session log: the CNN results, the plan and the uncommitted AI log changes.

<details><summary>6 commands</summary>

**Ran:** List session logs, plan tasks, and AI log diff

```bash
cd /home/amaragance/Desktop/Gerfigreynd-Project; ls -lt ~/.claude/projects/-home-amaragance-Desktop-Gerfigreynd-Project/*.jsonl | head -8; echo; cat docs/PROJECT_PLAN.md | sed -n '1,200p' | grep -n -i -E 'T1[0-9]|T2[0-9]|freeze|test' | head -40; echo; git diff --stat; git diff docs/AI_USAGE.md | head -120
```

**Ran:** Print CNN results notebook cells and outputs

```bash
cd /home/amaragance/Desktop/Gerfigreynd-Project; /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python - <<'EOF'
import json
nb=json.load(open('notebooks/02_cnn_results.ipynb'))
for i,c in enumerate(nb['cells']):
    src=''.join(c['source'])
    print(f'--- [{i}] {c["cell_type"]}')
    print(src[:1500])
    for o in c.get('outputs',[]):
        if 'text' in o: print('>>', ''.join(o['text'])[:1500])
        elif 'data' in o and 'text/plain' in o['data']: print('>>', ''.join(o['data']['text/plain'])[:1500])
EOF
```

**Ran:** Find where the handoff text appears in session logs

```bash
cd ~/.claude/projects/-home-amaragance-Desktop-Gerfigreynd-Project; for s in cc8994a0-13be-4156-afde-b0a4b1e5b002 376f460f-7079-4666-8d6e-70dc1363b6a4; do echo "=== $s"; grep -n -o 'Handoff: REI505M[^"]\{0,80\}' $s.jsonl | head -5; done
```

**Ran:** Print the full handoff text from the previous session

```bash
cd ~/.claude/projects/-home-amaragance-Desktop-Gerfigreynd-Project; /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python - <<'EOF'
import json
lines=open('cc8994a0-13be-4156-afde-b0a4b1e5b002.jsonl').read().splitlines()
for i in range(780, len(lines)):
    d=json.loads(lines[i])
    m=d.get('message',{})
    c=m.get('content')
    if isinstance(c,str): texts=[c]
    elif isinstance(c,list): texts=[x.get('text','') for x in c if x.get('type')=='text']
    else: texts=[]
    for t in texts:
        if 'Handoff: REI505M brain-age project, results' in t:
            print(i, d.get('type'), len(t)); print(t)
EOF
```

**Ran:** List transcripts and find exporter writes in session logs

```bash
cd ~/.claude/projects/-home-amaragance-Desktop-Gerfigreynd-Project; ls -la /home/amaragance/Desktop/Gerfigreynd-Project/docs/ai_transcripts/; /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python - <<'EOF'
import json,glob
for f in ['8811b7a3-1944-422a-b36b-7d422eb74514.jsonl','cc8994a0-13be-4156-afde-b0a4b1e5b002.jsonl']:
    for i,l in enumerate(open(f)):
        d=json.loads(l)
        c=d.get('message',{}).get('content')
        if not isinstance(c,list): continue
        for x in c:
            if x.get('type')=='tool_use' and 'export_transcript' in json.dumps(x.get('input',{})) and x['name'] in ('Write','Edit','Bash'):
                inp=x['input']
                print(f[:8], i, x['name'], (inp.get('file_path') or inp.get('command','')[:120]), d.get('timestamp'))
EOF
```

**Ran:** Show how the exporter was recreated last session

```bash
cd ~/.claude/projects/-home-amaragance-Desktop-Gerfigreynd-Project; /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python - <<'EOF'
import json
L=open('cc8994a0-13be-4156-afde-b0a4b1e5b002.jsonl').read().splitlines()
for i in (679,684,695):
    d=json.loads(L[i])
    for x in d['message']['content']:
        if x.get('type')=='tool_use': print(f'--- {i}'); print(x['input'].get('command'))
L=open('8811b7a3-1944-422a-b36b-7d422eb74514.jsonl').read().splitlines()
for i in (550,):
    d=json.loads(L[i])
    for x in d['message']['content']:
        if x.get('type')=='tool_use': print(f'--- 8811 {i}'); print(x['input'].get('command'))
EOF
```

</details>

Now I'll recreate the exporter the same way and export the five transcripts into the scratchpad. Nothing in the repo changes yet.

<details><summary>2 commands</summary>

**Ran:** Recreate transcript exporter in scratchpad

```bash
S=/tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/376f460f-7079-4666-8d6e-70dc1363b6a4/scratchpad/export; mkdir -p $S/transcripts && cd $S && /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python - <<'EOF'
import json
from pathlib import Path
lines = open("/home/amaragance/.claude/projects/-home-amaragance-Desktop-Gerfigreynd-Project/8811b7a3-1944-422a-b36b-7d422eb74514.jsonl").readlines()
for c in json.loads(lines[523])["message"]["content"]:
    if c.get("type") == "tool_use" and c["name"] == "Write":
        s = c["input"]["content"]
s = s.replace('''def local(ts):''', '''REDACT = {"[email]": "[email]", "[other project]": "[other project]", "[other project]": "[other project]",
          "[other project]": "[other project]", "[other project]": "[other project]", "[other project]": "[other project]",
          "[other project]": "[other project]"}


def local(ts):''', 1)
old = '''    Path(out).write_text("\\n".join(head + lines))'''
assert s.count(old) == 1
s = s.replace(old, '''    text = "\\n".join(head + lines)
    for word, repl in REDACT.items():
        text = text.replace(word, repl)
    Path(out).write_text(text)''')
Path("export_transcript.py").write_text(s)
EOF
sed -n '1,40p' export_transcript.py; grep -n "argv\|argparse\|--from\|--to" export_transcript.py | head
```

**Ran:** Show headers and endings of existing transcripts

```bash
cd /home/amaragance/Desktop/Gerfigreynd-Project/docs/ai_transcripts; for f in *.md; do echo "=== $f"; head -8 "$f"; echo ...; tail -3 "$f" | cut -c1-200; done
```

</details>

The exporter is back. Next I'll check what the two side sessions were about so their transcript files get accurate names.

<details><summary>2 commands</summary>

**Ran:** List the prompts and time span of the two side sessions

```bash
cd ~/.claude/projects/-home-amaragance-Desktop-Gerfigreynd-Project; for f in dd3686c9-e5c2-41ee-a780-aff405a5a312 2566d4f0-82d3-494d-bd7b-f160e144e3e5; do echo "== $f"; /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python - "$f" <<'EOF'
import json,sys,re
f=sys.argv[1]; ts=[]
for line in open(f+'.jsonl'):
    d=json.loads(line)
    if d.get('timestamp'): ts.append(d['timestamp'])
    if d.get('type')=='user' and not d.get('isMeta'):
        c=d['message']['content']
        if isinstance(c,list):
            if any(x.get('type')=='tool_result' for x in c if isinstance(x,dict)): continue
            t=' '.join(x.get('text','') for x in c if isinstance(x,dict) and x.get('type')=='text')
        else: t=c
        t=re.sub(r'<(system-reminder|ide_[a-z_]+)>.*?</\1>','',t,flags=re.S).strip()
        if t: print(d['timestamp'][:16], '|', t[:300].replace('\n',' '))
print('span', min(ts), max(ts))
EOF
done
```

**Ran:** Export five transcripts to scratchpad and check them

```bash
S=/tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/376f460f-7079-4666-8d6e-70dc1363b6a4/scratchpad/export; cd $S; L=~/.claude/projects/-home-amaragance-Desktop-Gerfigreynd-Project; PY=/home/amaragance/miniforge3/envs/rei505m-brainage/bin/python; T=transcripts
$PY export_transcript.py $L/8811b7a3-1944-422a-b36b-7d422eb74514.jsonl $T/2026-11-15_oli_claude-code_cnn.md "CNN loader, model and first training (12 and 15 Nov 2026)" && \
$PY export_transcript.py $L/cc8994a0-13be-4156-afde-b0a4b1e5b002.jsonl $T/2026-11-15_oli_claude-code_cnn-training.md "CNN training code, overfit check and first full run (15 Nov 2026)" --to 2026-11-15 && \
$PY export_transcript.py $L/cc8994a0-13be-4156-afde-b0a4b1e5b002.jsonl $T/2026-11-16_oli_claude-code_cnn-fix-and-comparison.md "Team questions, CNN batch-norm fix, comparison with the baseline and notebook troubleshooting (16 Nov 2026)" --from 2026-11-16 && \
$PY export_transcript.py $L/dd3686c9-e5c2-41ee-a780-aff405a5a312.jsonl $T/2026-11-15_oli_claude-code_ridge-question.md "What ridge means in this project (15 Nov 2026)" && \
$PY export_transcript.py $L/2566d4f0-82d3-494d-bd7b-f160e144e3e5.jsonl $T/2026-11-16_oli_claude-code_cnn-results-notebook.md "Implementing the CNN results notebook (16 Nov 2026)" && \
ls -la $T; for f in $T/*.md; do sed -n 3p $f; done
echo '--- privacy scan'; grep -c -i -E "[email]|@gmail|password|api[_-]?key|[other project]|[other project]|[other project]|[other project]|[other project]|[other project]|system-reminder|ide_opened" $T/*.md
grep -n -i -E "gh auth|\.ssh|credential|secret|\.netrc|bearer|token=" $T/*.md | cut -c1-160 | head
echo '--- diff vs repo copies (old line count / new line count)'; for f in 2026-11-15_oli_claude-code_cnn.md 2026-11-15_oli_claude-code_cnn-training.md 2026-11-16_oli_claude-code_cnn-fix-and-comparison.md; do echo "$f: $(wc -l < /home/amaragance/Desktop/Gerfigreynd-Project/docs/ai_transcripts/$f) -> $(wc -l < $T/$f); lines removed: $(diff /home/amaragance/Desktop/Gerfigreynd-Project/docs/ai_transcripts/$f $T/$f | grep -c '^<')"; done
```

</details>

Two things to check before I trust these: what the privacy-scan hits are, and why the re-export of the 12–15 Nov transcript drops 117 lines from the repo copy.

<details><summary>1 command</summary>

**Ran:** Inspect scan hits and differences from repo copies

```bash
S=/tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/376f460f-7079-4666-8d6e-70dc1363b6a4/scratchpad/export/transcripts; R=/home/amaragance/Desktop/Gerfigreynd-Project/docs/ai_transcripts
grep -n -o -i -E ".{0,60}([email]|@gmail|password|api[_-]?key|[other project]|[other project]|[other project]|[other project]|[other project]|[other project]|system-reminder|ide_opened).{0,40}" $S/2026-11-15_oli_claude-code_cnn.md $S/2026-11-16_oli_claude-code_cnn-fix-and-comparison.md
echo '--- cnn-fix removed lines'; diff $R/2026-11-16_oli_claude-code_cnn-fix-and-comparison.md $S/2026-11-16_oli_claude-code_cnn-fix-and-comparison.md | grep '^[<>]' | head -4 | cut -c1-200
echo '--- cnn.md removed lines (sample)'; diff $R/2026-11-15_oli_claude-code_cnn.md $S/2026-11-15_oli_claude-code_cnn.md | grep -n '^[<>-]\|^[0-9]' | head -30 | cut -c1-200
```

</details>

---

## Óli, 2026-11-16 18:42

*(sent while Claude was still working on the previous prompt)*

> # Handoff: REI505M brain-age project, results and report material (from session of 15–16 Nov 2026)
>
> ## Who and how to work (read first)
> - User: Óli (Ólafur Sær Sigursteinsson, GitHub Amaragance682), use they/them. Teammates: Sigga
>   (Sigríður María Jónsdóttir, macOS, uses Codex) and Amandine (Amandine Sophie S Thouvenot).
> - Óli does the hands-on work; you are an independent partner:
>   - Before creating or changing ANY file in the repo, show the proposed content in chat and wait for
>     a go-ahead. Óli types or applies it. Exception: transcripts Óli asks you to export.
>   - Never commit or push; Óli does all git.
>   - Docs are written as text in chat; Óli puts them in docs/.
>   - Don't finalise docs/PROJECT_GUIDE.md. Ignore the global PRESENTATION_TOPICS.md rule here.
> - Honor code: AI must not write any part of the report, and figure and table captions count as report text.
>   - For each section, give ingredients: points to make, numbers, which table or figure, citations
>     to look up. Then review and edit what Óli writes.
>   - Example sentences are allowed by Óli's rule, but any that are used must be declared in
>     docs/AI_USAGE.md.
>   - Code that produces tables and figures is fine to help with, marked as AI-assisted.
> - Test every snippet in your scratchpad on the real data, against Óli's actual files, before showing it.
> - Memory files load automatically, including the transcript-export convention.
>
> ## Deadlines (today is 16 Nov)
> - Plan (docs/PROJECT_PLAN.md):
>   - T14 freeze settings and T15 test comparison: 16 Nov (not done yet).
>   - T16 figures and interpretation: 18 Nov.
>   - T18 package: 19 Nov.
> - Final report (PDF, max 10 pages excluding appendices, past tense, no code in the text) plus all
>   code, zipped and emailed to steinng@hi.is by Fri 20 Nov.
> - Report sections required by the assignment (REI505M_Final_project.pdf, section 2):
>   - Introduction (0.5–1 page: what, why, how) and prior work (with citations);
>   - AI statement; dataset and processing;
>   - methods (algorithms, why they were chosen, how they were evaluated, how models and
>     hyperparameters were selected);
>   - results (tables and figures with captions, referred to in the text; effects of hyperparameter
>     choice; training/validation curves; experiments that did not work; text in figures not too small);
>   - conclusions and future work; references; collaboration (who did what);
>   - appendix documenting AI use.
>
> ## Machine and environment
> - Repo /home/amaragance/Desktop/Gerfigreynd-Project (private GitHub, branch main). Editor: Code - OSS.
> - Python: /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python (torch 2.14.1+cu130).
>   - After a reboot, Óli's fish terminal needs `conda activate rei505m-brainage`; plain `python` has
>     no pandas.
>   - Use `-u` when writing to a log.
> - Notebooks: use plain `from tqdm import tqdm`. `tqdm.auto` draws a widget, and Code - OSS then stops
>   sending later cells; the cell looks "running" forever. Headless fallback:
>   `jupyter nbconvert --to notebook --execute --inplace notebooks/<nb>.ipynb`.
> - GPU: RTX 5080 Laptop, 16 GB. Óli sometimes runs games that take ~7 GB; it doesn't change results
>   but slows runs. Ask Óli to close them before real runs.
> - /tmp (your scratchpad) is wiped on every reboot; it happened twice on 16 Nov. Nothing durable
>   goes there.
>
> ## Data (instructor-preprocessed; unchanged)
> - data/pre_processed/<ID>.pt: float16 (20, 57664) = 20 bipolar EEG channels x 901 s at 64 Hz.
>   - Band-passed 0.5–30 Hz; cut every file at 901 s (one is 948 s).
>   - 0–300 s is pre-N2, 300–900 s is N2.
>   - No ECG channel in these files.
> - Splits: data/segments_h1k_{train,val,test}.pt, 825 / 103 / 104 children, made by the
>   instructor. No id appears in two splits; the label file has no subject id, so repeat recordings
>   of one child can't be checked.
> - Labels: data/anon_codex_age_sex.csv. Ages 0.16–16, rounded (spacing ~16 days for infants to
>   ~150 days for teens).
>   - Median age: train 2.98, val 2.07, test 2.63.
>   - Under 6: train 567, val 80, test 74.
>   - 15 or over: 9 / 1 / 1.
> - Age groups everywhere: <1, 1–3, 3–6, 6–12, 12+ (left-closed). Val counts 30/32/18/16/7.
>
> ## Code and outputs now (all run by Óli from the repo)
> - notebooks/01_baseline.ipynb:
>   - 13 features per channel (log variance, Hjorth mobility and complexity, log and relative power
>     in delta, theta, alpha, sigma, beta) from 30 s windows;
>   - median over windows, separately for pre-N2 and N2: 520 features;
>   - StandardScaler + RidgeCV on log(age), alpha by leave-one-out on train (picks 56);
>   - writes outputs/features_win30.csv and outputs/ridge/val_predictions.csv (id, age, pred).
>   - Validation table:
>
>     | model | MAE | balanced MAE |
>     |---|---|---|
>     | predict the training median | 2.98 | 4.37 |
>     | ridge, C3-P3 only | 1.03 | 1.34 |
>     | ridge, all channels (THE baseline) | 0.75 | 1.08 |
>     | ridge, pre-N2 only | 1.24 | 1.94 |
>     | ridge, N2 only | 0.78 | 1.29 |
>
> - scripts/train_cnn.py (flags `--check`, `--overfit`, `--train`, `--epochs`, `--run-name`):
>   - Model: figure 3 of the assignment. 3 ConvBlocks of [Conv2d(1,3)+BN+ReLU] x2 + MaxPool(1,2),
>     widths 16/32/64. Then the "fully connected" Conv2d(64, 64, (1,128)) + BatchNorm2d + ReLU,
>     max over the 20 channels → z (64), Dropout(0.3) + Linear → log(age). About 549k parameters.
>     The model gives the same answer if the channels are reordered.
>   - The BatchNorm2d after the fc conv was added on 16 Nov (commit 66dc232). Without it the fc
>     ReLUs died: 11 of 64 features alive, and a run on under-6s collapsed to a constant
>     prediction. With it, all 64 stay alive.
>   - Training:
>     - 16 s random crops inside the real signal, 8 per child per epoch, batch 64;
>     - AdamW (lr 1e-3, weight decay 1e-2), cosine schedule over a fixed 60 epochs, Huber loss on
>       log(age), head bias set to the mean log(age), seed 0;
>     - the FINAL model is kept (no best-epoch selection);
>     - batch-norm statistics are recomputed over one fixed epoch of training crops before every
>       validation and before saving.
>   - Prediction: eval mode, all non-overlapping 16 s windows per child, median log(age) → years.
>   - `--overfit`: 20 children across the age range, one fixed 16 s window each, no regularisation.
>     It passes: MAE 0.018 years.
>   - outputs/cnn16s/: final.pt (state_dict, scale, settings), learning_curve.csv,
>     val_predictions.csv. Full run takes 5.1 min.
>   - The same code and seed varies by about 0.02 MAE between runs (GPU nondeterminism).
> - notebooks/02_cnn_results.ipynb (6 cells; implemented with help from another Claude session):
>   - setup and evaluate();
>   - learning-curve figure (loss, val MAE and balanced MAE, lr) → outputs/figures/cnn_learning_curve.png;
>   - joins CNN and ridge predictions on id;
>   - comparison table;
>   - stratified paired bootstrap (10,000 resamples within age groups, seed 0);
>   - 3-panel figure (predicted vs. true, residuals, MAE per group) → outputs/figures/cnn_vs_ridge_val.png.
>   - Ridge is blue (#2a78d6) and the CNN orange (#eb6834) in every figure.
>   - No cell yet reloads final.pt and re-predicts; the test step needs one (BrainAgeCNN,
>     load_split, eval_windows, predict_children with ck["scale"]).
> - Validation results (Óli's runs):
>
>   | val (103 children) | MAE | balanced | <1 | 1–3 | 3–6 | 6–12 | 12+ | r | R² |
>   |---|---|---|---|---|---|---|---|---|---|
>   | ridge | 0.75 | 1.08 | 0.13 | 0.49 | 0.75 | 1.77 | 2.26 | 0.94 | 0.89 |
>   | CNN | 0.60 | 0.89 | 0.10 | 0.33 | 0.88 | 1.18 | 1.97 | 0.97 | 0.93 |
>
>   - Bootstrap, CNN − ridge:
>     - MAE −0.148, 95% interval [−0.340, +0.039]; the CNN is worse in 5.9% of resamples;
>     - balanced MAE −0.188, 95% interval [−0.603, +0.210]; 18.7%.
>   - So the CNN is likely better, but on val that isn't certain.
>   - Both models predict teenagers too young.
> - outputs/ is git-ignored. REPORT_DATA/ at the repo root is a new, empty folder for report-ready
>   material; Óli will say how to lay it out.
>
> ## Scratch results (AI-run on 16 Nov; NOT for the report until rerun from repo code)
> - CNN before the BN fix (same seed): 0.68 / 1.05.
> - CNN on N2 only: 0.84 / 1.37. It is worse mainly for 12+ (3.79), as with ridge N2-only (3.62),
>   so pre-N2 adds information.
> - Specialising on under-6s doesn't help. On the same 80 val children under 6:
>   - ridge trained on all ages 0.415, trained on under-6 only 0.440;
>   - CNN (with the BN fix) 0.365 vs. 0.449.
> - One-channel N2 ridge (13 features): best T6-O2 0.98; C3-CZ ranks 16th of 20 (2.17).
> - The two models' errors correlate only 0.25; averaging their predictions gives 0.55 / 0.77 on val.
>   That is an observation for future work, not a model to add now.
> - The 23 val children aged 6+ produce 57% of ridge's summed error. Median relative error is ~16%
>   for both models.
>
> ## Goals for this session, in order
> 1. Walk through 02_cnn_results.ipynb with Óli, cell by cell. Check the figures for report use: the
>    3-panel figures are 14–16 inches wide, so text becomes tiny at page width. Plan report versions
>    (larger fonts, fewer panels per figure, PDF for LaTeX or 300 dpi PNG for Word).
> 2. Agree a REPORT_DATA layout with Óli, e.g. figures/, tables/ (CSV, plus Markdown or LaTeX) and a
>    README index: file → which notebook or script makes it → which report section. The notebooks
>    should save there directly, so everything stays reproducible.
> 3. Make the settings experiments reproducible from the repo, for "effects of hyperparameter choice"
>    and "experiments that did not work".
>    - Propose small train_cnn.py flags, e.g. `--n2-only`, `--max-age 6`, `--no-fc-bn`, `--seed`.
>    - Óli reruns: BN fix vs. no fix, N2-only, under-6; optionally MSE on age vs. Huber on log(age),
>      which the plan promises.
>    - Validation only.
> 4. Freeze settings (T14) and decide the whole test protocol BEFORE touching test; then run it once
>    (T15).
>    - CNN with the current settings, trained on train only, for 3 seeds. Report each seed and the
>      spread, and decide beforehand how the bootstrap uses them.
>    - Ridge (all channels) fitted on train only; add test-prediction saving to 01 in the same
>      id, age, pred format.
>    - Same table, bootstrap and figures as for val.
> 5. Report map: for each section above, list ingredients, numbers, tables and figures, and
>    citations. Include Iyer et al. 2024 and Stevenson et al. 2026 from the assignment; Óli's team
>    reads them. No prose. Put the final tables and figures into REPORT_DATA.
> 6. End the session with:
>    - what else could be done (future work);
>    - how to handle the ECG model (below);
>    - AI_USAGE.md entry text in chat;
>    - a transcript export.
>
> ## ECG model (a teammate's plan) — advice so far
> - The ECG is not in the preprocessed files. It must come from the raw EDF files (Zenodo, 6 GB),
>   preprocessed by the teammate and cut to the same 0–901 s.
> - Keep it in a separate script or notebook that imports helpers from train_cnn.py (eval_windows,
>   predict_children, mae_scores, EDGES, GROUPS) rather than editing train_cnn.py, which is behind the
>   main results. If shared code changes, rerun `--check` and `--overfit`.
> - Don't add it as a 21st channel to this CNN: shared filters, the max over channels and the EEG
>   scale (42 µV) would let the ~1 mV ECG dominate.
> - Use the same splits and the same id, age, pred files, so it fits the comparison table and
>   bootstrap.
> - Heart rate falls steeply with age in children, so a heart-rate-feature baseline may do well.
>   Frame it as age predicted from the heart, as an extension; "future work" if time runs out.
>
> ## Decisions already made
> - Keep all ages 0–16; no 0–15 cut and no under-6-only model.
> - Ridge on all channels stays the baseline.
> - Spindle and sigma-peak features (a teammate's plan for C3-CZ in N2) only after the report
>   essentials. If added, add them as extra baseline columns or a separate small row, never as the
>   main baseline.
>
> ## Housekeeping
> - Export transcripts. The exporter is not in the repo; recover it from the session log as the
>   memory file describes.
>   - First, re-export this session (cc8994a0-13be-4156-afde-b0a4b1e5b002.jsonl), so the copies in
>     docs/ai_transcripts/ include its end:
>     - --to 2026-11-15 → 2026-11-15_oli_claude-code_cnn-training.md,
>       "CNN training code, overfit check and first full run (15 Nov 2026)";
>     - --from 2026-11-16 → 2026-11-16_oli_claude-code_cnn-fix-and-comparison.md,
>       "Team questions, CNN batch-norm fix, comparison with the baseline and notebook troubleshooting (16 Nov 2026)".
>   - Export the side sessions: dd3686c9 (15 Nov, "what is ridge…") and 2566d4f0 (16 Nov, helped
>     implement 02_cnn_results).
>   - Re-export 8811b7a3 (its docs copy stops before its handoff).
> - docs/AI_USAGE.md:
>   - the entries for 15 Nov evening and 16 Nov were drafted; check that Óli added them;
>   - the existing 15 Nov entry needs a **Transcript:** line;
>   - entries are still needed for the side sessions;
>   - the 02 notebook needs an AI-assisted mark.
> - Tell the team:
>   - T12 saves the final checkpoint, not the best validation one;
>   - the BN fix;
>   - the teammate's spindle document says ChatGPT wrote a feature script and a results summary.
>     Both need documenting, and rewording AI-written text still counts as AI-written report text.
> - PROJECT_PLAN says "average predictions per recording"; we use the median. The definitions doc
>   (T07) should match.
> - Older items from Sigga's push: data_audit.ipynb bugs, the capital S in "Segments_", and the T07
>   definitions conflict.

**Claude:**

<details><summary>8 commands</summary>

**Ran:** Compare the changed block in the old and new transcript

```bash
S=/tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/376f460f-7079-4666-8d6e-70dc1363b6a4/scratchpad/export/transcripts; R=/home/amaragance/Desktop/Gerfigreynd-Project/docs/ai_transcripts
echo '--- OLD 515-530'; sed -n '515,530p' $R/2026-11-15_oli_claude-code_cnn.md | cut -c1-160
echo '--- NEW 515-540'; sed -n '515,540p' $S/2026-11-15_oli_claude-code_cnn.md | cut -c1-160
echo '--- OLD 630-645'; sed -n '630,645p' $R/2026-11-15_oli_claude-code_cnn.md | cut -c1-160
echo '--- NEW 650-668'; sed -n '650,668p' $S/2026-11-15_oli_claude-code_cnn.md | cut -c1-160
diff $R/2026-11-15_oli_claude-code_cnn.md $S/2026-11-15_oli_claude-code_cnn.md | grep '^[0-9]'
```

**Ran:** Check whether the session log holds two versions of the message

```bash
S=/tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/376f460f-7079-4666-8d6e-70dc1363b6a4/scratchpad/export; grep -c "FUNCTION load_split" $S/transcripts/2026-11-15_oli_claude-code_cnn.md /home/amaragance/Desktop/Gerfigreynd-Project/docs/ai_transcripts/2026-11-15_oli_claude-code_cnn.md; grep -n "parentUuid\|isSidechain\|def load\|for line in\|uuid" $S/export_transcript.py | head; cd ~/.claude/projects/-home-amaragance-Desktop-Gerfigreynd-Project; /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python - <<'EOF'
import json
L=[json.loads(l) for l in open('8811b7a3-1944-422a-b36b-7d422eb74514.jsonl')]
hits=[]
for i,d in enumerate(L):
    c=d.get('message',{}).get('content')
    if d.get('type')=='assistant' and isinstance(c,list):
        for x in c:
            if x.get('type')=='text' and 'Proposed `scripts/train_cnn.py` (step 1)' in x['text']:
                hits.append(i); print(i, d.get('uuid'), d.get('parentUuid'), d.get('timestamp'), 'FUNCTION load_split' in x['text'], 'def load_split' in x['text'])
    if d.get('type') in ('summary','file-history-snapshot') : pass
# check for multiple children of same parent (branches)
from collections import Counter
par=Counter(d.get('parentUuid') for d in L if d.get('type') in ('user','assistant'))
print('branch points:', sum(1 for k,v in par.items() if v>1 and k))
print({d.get('type') for d in L})
EOF
```

**Ran:** Search all logs for the pseudocode version's origin

```bash
cd ~/.claude/projects/-home-amaragance-Desktop-Gerfigreynd-Project; /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python - <<'EOF'
import json,glob
for f in sorted(glob.glob('*.jsonl')):
    for i,l in enumerate(open(f)):
        if 'FUNCTION load_split' in l:
            d=json.loads(l)
            c=d.get('message',{}).get('content')
            kind=[x.get('type')+':'+x.get('name','') for x in c] if isinstance(c,list) else 'str'
            print(f[:8], i, d.get('type'), d.get('timestamp'), kind)
EOF
cd /home/amaragance/Desktop/Gerfigreynd-Project; git log --oneline --follow -- docs/ai_transcripts/2026-11-15_oli_claude-code_cnn.md; git diff --stat HEAD -- docs/ai_transcripts/2026-11-15_oli_claude-code_cnn.md
```

**Ran:** Show the remaining differences in the 12–15 Nov transcript

```bash
S=/tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/376f460f-7079-4666-8d6e-70dc1363b6a4/scratchpad/export/transcripts; R=/home/amaragance/Desktop/Gerfigreynd-Project/docs/ai_transcripts; diff $R/2026-11-15_oli_claude-code_cnn.md $S/2026-11-15_oli_claude-code_cnn.md | awk '/^526,638c/{skip=1} /^665a/{skip=0} /^2151a/{exit} !skip' | cut -c1-220 | head -120
```

**Ran:** Search logs for the prompt wordings found only in the repo copy

```bash
cd ~/.claude/projects/-home-amaragance-Desktop-Gerfigreynd-Project; /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python - <<'EOF'
import json,glob
needles=["what items 100% need to be on the report","dont quite get how they work and relate","A decision for you before we start the training code","before we move on to the training code, can we fill in"]
for f in sorted(glob.glob('*.jsonl')):
    if f.startswith('376f460f'): continue
    for i,l in enumerate(open(f)):
        for n in needles:
            if n in l:
                d=json.loads(l); print(f[:8], i, d.get('type'), d.get('timestamp'), d.get('uuid','')[:8], 'parent', (d.get('parentUuid') or '')[:8], '|', n[:40])
EOF
```

**Ran:** Re-export older transcripts to check for edits; show scan hits

```bash
S=/tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/376f460f-7079-4666-8d6e-70dc1363b6a4/scratchpad/export; R=/home/amaragance/Desktop/Gerfigreynd-Project/docs/ai_transcripts; L=~/.claude/projects/-home-amaragance-Desktop-Gerfigreynd-Project; PY=/home/amaragance/miniforge3/envs/rei505m-brainage/bin/python; mkdir -p $S/check; cd $S
$PY export_transcript.py $L/4652e401-2e26-46fd-a580-d00f23cb9b3f.jsonl check/setup.md "Environment setup and project overview (2 Nov 2026)" --to 2026-11-02 >/dev/null
$PY export_transcript.py $L/4652e401-2e26-46fd-a580-d00f23cb9b3f.jsonl check/baseline.md "Baseline model, interim report and review of teammate changes (11-12 Nov 2026)" --from 2026-11-03 >/dev/null
for p in "setup.md 2026-11-02_oli_claude-code_setup.md" "baseline.md 2026-11-11_oli_claude-code_baseline.md"; do set -- $p; echo "== $2: repo $(wc -l < $R/$2) lines, fresh $(wc -l < check/$1); diff hunks: $(diff $R/$2 check/$1 | grep -c '^[0-9]')"; sed -n 3p check/$1; diff $R/$2 check/$1 | grep '^[<>]' | head -6 | cut -c1-160; done
echo '== scan hits in cnn-fix'; grep -n -i -E "[email]|@gmail|password|api[_-]?key|[other project]|[other project]|[other project]|[other project]|[other project]|[other project]|system-reminder|ide_opened" transcripts/2026-11-16_oli_claude-code_cnn-fix-and-comparison.md | cut -c1-200
```

**Ran:** Read the two new side-session transcripts

```bash
S=/tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/376f460f-7079-4666-8d6e-70dc1363b6a4/scratchpad/export/transcripts; cat $S/2026-11-15_oli_claude-code_ridge-question.md | cut -c1-300; echo; echo '======== results-notebook (prose only)'; grep -v '^\s*$' $S/2026-11-16_oli_claude-code_cnn-results-notebook.md | awk '/<details>/{d=1} !d{print} /<\/details>/{d=0}' | cut -c1-250 | head -90
```

**Ran:** Compare older committed transcripts with fresh exports

```bash
S=/tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/376f460f-7079-4666-8d6e-70dc1363b6a4/scratchpad/export; R=/home/amaragance/Desktop/Gerfigreynd-Project/docs/ai_transcripts; cd $S; ls check
chk() { echo "== $2: repo $(wc -l < $R/$2) lines, fresh $(wc -l < check/$1); diff hunks: $(diff $R/$2 check/$1 | grep -c '^[0-9]')"; sed -n 3p check/$1; diff $R/$2 check/$1 | grep '^[<>]' | head -8 | cut -c1-180; }
chk setup.md 2026-11-02_oli_claude-code_setup.md
chk baseline.md 2026-11-11_oli_claude-code_baseline.md
```

</details>

The setup transcript matches its log exactly. The baseline transcript does not: the repo copy is missing 162 lines that are in the log. I'll check whether those are whole exchanges.

<details><summary>4 commands, 2 files read</summary>

**Ran:** List the exchanges missing from the baseline transcript

```bash
S=/tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/376f460f-7079-4666-8d6e-70dc1363b6a4/scratchpad/export; R=/home/amaragance/Desktop/Gerfigreynd-Project/docs/ai_transcripts; diff $R/2026-11-11_oli_claude-code_baseline.md $S/check/baseline.md | grep -E '^[0-9]|^> ## |^> > |^< ' | cut -c1-230; cd /home/amaragance/Desktop/Gerfigreynd-Project; git log --format='%h %ad %s' --date=short -- docs/ai_transcripts/2026-11-11_oli_claude-code_baseline.md docs/ai_transcripts/2026-11-15_oli_claude-code_cnn.md
```

**Ran:** Search prompt history and compare transcript file times

```bash
cd ~/.claude; grep -rl --include='*.jsonl' -F "what items 100% need to be on the report" . 2>/dev/null | head; grep -rl -F "i dont think anything else can be fitted" . 2>/dev/null | grep -v 376f460f | head; echo '--- mtimes'; stat -c '%y %n' /home/amaragance/Desktop/Gerfigreynd-Project/docs/ai_transcripts/*.md; cd /home/amaragance/Desktop/Gerfigreynd-Project; git show --stat --format='%h %ad' --date=iso b9aec92 | head -20
```

**Ran:** Read missing 11 Nov exchanges and the AI log so far

```bash
B=/tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/376f460f-7079-4666-8d6e-70dc1363b6a4/scratchpad/export/check/baseline.md; awk 'NR>=677&&NR<=726 || NR>=923&&NR<=950 || NR>=1006&&NR<=1064 || NR>=1136&&NR<=1160' $B | awk '/<details>/{d=1} !d{print} /<\/details>/{d=0}' | grep -v '^\s*$' | cut -c1-260; echo; echo '======== AI_USAGE (committed part up to 15 Nov)'; sed -n '1,118p' /home/amaragance/Desktop/Gerfigreynd-Project/docs/AI_USAGE.md | cut -c1-220
```

**Ran:** Copy the uncontested transcripts into the repo

```bash
S=/tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/376f460f-7079-4666-8d6e-70dc1363b6a4/scratchpad/export/transcripts; R=/home/amaragance/Desktop/Gerfigreynd-Project/docs/ai_transcripts; cmp $S/2026-11-15_oli_claude-code_cnn-training.md $R/2026-11-15_oli_claude-code_cnn-training.md && echo "cnn-training: unchanged, already up to date"; cp $S/2026-11-16_oli_claude-code_cnn-fix-and-comparison.md $S/2026-11-15_oli_claude-code_ridge-question.md $S/2026-11-16_oli_claude-code_cnn-results-notebook.md $R/ && cd /home/amaragance/Desktop/Gerfigreynd-Project && git status --short docs/
```

**Read** `outputs/figures/cnn_vs_ridge_val.png`

**Read** `outputs/figures/cnn_learning_curve.png`

</details>

Your first paste stopped after the baseline table. The full handoff you re-sent matches what I recovered from the last session's log, so I'm working from that.

## Transcripts: four ready, two need your decision

I exported these with the recovered exporter. The privacy scan is clean; its only hits were the scan command itself.
- [2026-11-16_oli_claude-code_cnn-fix-and-comparison.md](docs/ai_transcripts/2026-11-16_oli_claude-code_cnn-fix-and-comparison.md) now runs to 18:41, including the handoff.
- Two new files cover the side sessions: [ridge-question](docs/ai_transcripts/2026-11-15_oli_claude-code_ridge-question.md) (15 Nov) and [cnn-results-notebook](docs/ai_transcripts/2026-11-16_oli_claude-code_cnn-results-notebook.md) (16 Nov).
- `cnn-training` was already complete.

I didn't overwrite two committed transcripts, because they no longer match the session logs. The setup transcript, exported at the same moment (19:48 on 15 Nov), still matches its log line for line. These two were changed afterwards, at 20:19 and 20:44, shortly before commit b9aec92:
- **[2026-11-15_oli_claude-code_cnn.md](docs/ai_transcripts/2026-11-15_oli_claude-code_cnn.md):**
  - The Python I wrote for `train_cnn.py` step 1 has been replaced with pseudocode.
  - Three of my replies are missing, including "the AI wrote ConvBlock, BrainAgeCNN and check_model" and the AI_USAGE draft.
  - Two of your prompts have text that isn't in the log.
- **[2026-11-11_oli_claude-code_baseline.md](docs/ai_transcripts/2026-11-11_oli_claude-code_baseline.md):**
  - Four exchanges from 11 Nov are missing:
    - the problems table for the interim report;
    - advice on what to put in the report;
    - a pathlib fix;
    - the reply where I drafted two sentences ("NEW 1" and "NEW 2") for the methods paragraph.
  - Five prompts have extra text.

The changed wording appears in no Claude log or prompt history, and both headers still say the prompts and replies are verbatim. There may be a good reason for the edits. Even so, as they stand the appendix shows less AI help than there was, and that is exactly what the honor code checks. What I'd do:
1. **Replace both files with the unedited exports.** They're ready, and the current versions stay in git history. If something has to come out, such as a remark about a teammate, cut it visibly with `[removed: …]` and say so in AI_USAGE.
2. **Make [AI_USAGE.md](docs/AI_USAGE.md) match the transcripts:**
   - 11–12 Nov: did the "NEW 1" and "NEW 2" sentences go into the submitted interim report? If so, declare them as AI-drafted sentences. Declare the problems-table notes too.
   - 15 Nov: "Helped develop the ConvBlock and BrainAgeCNN classes" should say that I wrote them and the loader functions, and you typed them in. This entry also needs its **Transcript:** line.
   - Add entries for the two side sessions, and an AI-assisted mark in `02_cnn_results`.

## A correction to the handoff
"Both models predict teenagers too young" is only half right on val:
- **Ridge, 12+:** mean residual +0.32 years (median +0.08), with large errors in both directions.
- **CNN, 12+:** mean residual −1.38 (median −0.33), mostly from three children predicted 3.3–4.1 years too young.
- **CNN, 3–6:** also predicts too young (−0.43). This is the only group where the CNN does worse than ridge.

## 02_cnn_results.ipynb
All six cells are correct. The join is on id, the age check uses a tolerance, the bootstrap resamples within age groups, and the observed differences are computed before rounding. For the report versions of the figures:
- **Size:** at 14–16 in wide, shrunk to a ~6.5 in text column, the labels end up around 4.5 pt. Make report versions about 7 in wide with 9–10 pt text, saved as PDF for LaTeX or 300 dpi PNG for Word.
- **Learning curve:** drop the learning-rate panel; one sentence in methods covers it. Training loss is Huber on log(age) while validation is MAE in years, so the curves aren't on the same scale. Logging validation loss as well would be a one-line change in `train_cnn.py`, and the coming reruns would pick it up for free.
- **Comparison figure:** the bar chart repeats the table, so keep predicted-vs-true and residuals as a 2-panel figure. Half the children are under 3 and pile up in one corner; a log-scale age axis would spread them out.

## Order for tonight
T14 and T15 (freeze settings and run the test comparison) are due today. The settings are effectively decided: all ages, the current CNN, ridge on all channels. The experiment reruns only use validation and won't change that. So I suggest:
1. Settle the transcripts and commit.
2. Write the test protocol down: 3 CNN seeds, how the bootstrap uses them, and saving ridge's test predictions from 01.
3. Run the test comparison once, with games closed. That's about 15 minutes of GPU time.
4. Tomorrow: the REPORT_DATA layout, experiment flags and report figures (T16 is due 18 Nov).

Shall I replace the two transcripts with the unedited exports, and does that order work for you?

---

## Óli, 2026-11-16 18:55

> sounds good, lets proceed with the freeze settings. 
>
> 1. done, ill do this at the end of the session.
>
> 2. sounds good, lets do this first
>
> 3. sounds good
> 4. lets do this tonight
>
> no need on the exports,ill handle it later

**Claude:**

Freezing first, then. I'll start by reading the current training script and baseline notebook, so the frozen settings and test protocol match what the code actually does.

<details><summary>7 commands</summary>

**Ran:** Read the CNN training script

```bash
cd /home/amaragance/Desktop/Gerfigreynd-Project; wc -l scripts/train_cnn.py; cat -n scripts/train_cnn.py
```

**Ran:** Read the baseline notebook and the feature file

```bash
cd /home/amaragance/Desktop/Gerfigreynd-Project; /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python - <<'EOF'
import json
nb=json.load(open('notebooks/01_baseline.ipynb'))
for i,c in enumerate(nb['cells']):
    src=''.join(c['source'])
    print(f'--- [{i}] {c["cell_type"]}')
    print(src[:2500])
    for o in c.get('outputs',[]):
        if 'text' in o: print('>>', ''.join(o['text'])[:800])
        elif 'data' in o and 'text/plain' in o['data']: print('>>', ''.join(o['data']['text/plain'])[:800])
EOF
head -c 300 outputs/features_win30.csv; echo; /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python -c "
import pandas as pd; f=pd.read_csv('outputs/features_win30.csv'); print(f.shape); print(f.iloc[:,:4].head(3)); print([c for c in f.columns if not c.startswith(('pre','n2','N2','PRE'))][:10])"
```

**Ran:** Check GPU usage and age-group counts per split

```bash
cd /home/amaragance/Desktop/Gerfigreynd-Project; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python - <<'EOF'
import torch, pandas as pd, numpy as np
EDGES=[0,1,3,6,12,np.inf]; GROUPS=["<1","1-3","3-6","6-12","12+"]
for s in ("train","val","test"):
    seg=torch.load(f"data/segments_h1k_{s}.pt",weights_only=True)
    a=pd.Series([t[3] for t in seg])
    print(s, len(a), pd.cut(a,EDGES,right=False,labels=GROUPS).value_counts().sort_index().to_dict())
EOF
```

**Ran:** Apply proposed changes to a scratch copy of the training script

```bash
SP=/tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/376f460f-7079-4666-8d6e-70dc1363b6a4/scratchpad/freeze; mkdir -p $SP/scripts $SP/outputs/cnn16s && cp /home/amaragance/Desktop/Gerfigreynd-Project/scripts/train_cnn.py $SP/scripts/train_cnn_repo.py && cp /home/amaragance/Desktop/Gerfigreynd-Project/outputs/cnn16s/{final.pt,val_predictions.csv} $SP/outputs/cnn16s/ && cd $SP && /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python - <<'EOF'
from pathlib import Path
s = Path("scripts/train_cnn_repo.py").read_text()
def rep(old, new):
    global s
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)

# A: split predict_children so training can reuse the window predictions for a val loss
rep('''def predict_children(model, windows, owner, scale, n_children):
    # every window of every child -> median of the predicted log(age) per child -> years
    model.eval()  # always predict in eval mode; train mode normalises each child's amplitude away
    with torch.no_grad():
        log_pred = torch.cat([model(w.float() / scale) for w in windows.split(512)])
    medians = torch.stack([log_pred[owner == i].quantile(0.5) for i in range(n_children)])
    return medians.exp().cpu().numpy()
''', '''def window_log_ages(model, windows, scale):
    # predicted log(age) of every window
    model.eval()  # always predict in eval mode; train mode normalises each child's amplitude away
    with torch.no_grad():
        return torch.cat([model(w.float() / scale) for w in windows.split(512)])


def child_ages(log_pred, owner, n_children):
    # median of the predicted log(age) over each child's windows -> years
    medians = torch.stack([log_pred[owner == i].quantile(0.5) for i in range(n_children)])
    return medians.exp().cpu().numpy()


def predict_children(model, windows, owner, scale, n_children):
    # every window of every child -> median of the predicted log(age) per child -> years
    return child_ages(window_log_ages(model, windows, scale), owner, n_children)
''')

# B: val loss in the learning curve, on the same scale as the training loss
rep('''        recompute_bn(model, X, lengths, scale, crops_per_child, batch_size, seed)
        pred = predict_children(model, windows, owner, scale, len(val_ids))
        mae, bal = mae_scores(val_age, pred)
        log.append({"epoch": epoch, "train_loss": np.mean(losses), "val_MAE": mae,
                    "val_balanced_MAE": bal, "lr": lr_used, "seconds": time.time() - t0})''',
'''        recompute_bn(model, X, lengths, scale, crops_per_child, batch_size, seed)
        log_pred = window_log_ages(model, windows, scale)
        val_loss = nn.functional.huber_loss(log_pred, torch.log(val_ages[owner])).item()  # same loss as training, per window
        pred = child_ages(log_pred, owner, len(val_ids))
        mae, bal = mae_scores(val_age, pred)
        log.append({"epoch": epoch, "train_loss": np.mean(losses), "val_loss": val_loss, "val_MAE": mae,
                    "val_balanced_MAE": bal, "lr": lr_used, "seconds": time.time() - t0})''')
rep('''        print(f"epoch {epoch:3d}  train loss {np.mean(losses):.4f}  val MAE {mae:.2f}  "''',
    '''        print(f"epoch {epoch:3d}  train loss {np.mean(losses):.4f}  val loss {val_loss:.4f}  val MAE {mae:.2f}  "''')

# C: predict test from a saved model, after checking it reproduces its val predictions
rep('''    print(f"final val MAE {mae:.2f}, balanced {bal:.2f}; saved to {out}")
''', '''    print(f"final val MAE {mae:.2f}, balanced {bal:.2f}; saved to {out}")


def predict_test(device, run_name):
    # T15: run once per frozen model, after the freeze (docs/FROZEN_SETTINGS.md). Nothing is trained or tuned here.
    # The reloaded model must first reproduce the val predictions saved at the end of training.
    run = OUTPUTS / run_name
    ck = torch.load(run / "final.pt", map_location=device, weights_only=True)
… (34 more lines)
```

**Ran:** Test the proposed script changes in the scratchpad

```bash
SP=/tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/376f460f-7079-4666-8d6e-70dc1363b6a4/scratchpad/freeze; cd $SP; PY=/home/amaragance/miniforge3/envs/rei505m-brainage/bin/python
$PY -u scripts/train_cnn_scratch.py --train --epochs 1 --seed 0 --run-name s0 2>&1 | tail -2
$PY -u scripts/train_cnn_scratch.py --train --epochs 1 --seed 1 --run-name s1 2>&1 | tail -2
echo '--- repo run cnn16s, epoch 1:'; head -2 /home/amaragance/Desktop/Gerfigreynd-Project/outputs/cnn16s/learning_curve.csv
echo '--- new curve columns:'; cat outputs/s0/learning_curve.csv
echo '--- predict_test, with "test" swapped for "val":'
$PY - <<'EOF'
import sys, numpy as np, pandas as pd, torch
sys.path.insert(0, "scripts")
import train_cnn_scratch as t
real = t.load_split
t.load_split = lambda split, device: real("val", device)   # stand-in: never load test in a dry run
t.predict_test(torch.device("cuda"), "cnn16s")
a = pd.read_csv("outputs/cnn16s/val_predictions.csv"); b = pd.read_csv("outputs/cnn16s/test_predictions.csv")
print("stand-in file matches val predictions:", (a.id == b.id).all(), np.abs(a.pred - b.pred).max(), list(b.columns))
EOF
echo '--- a model that does not match its val file must fail:'
$PY - <<'EOF' 2>&1 | tail -1
import sys, pandas as pd, torch
sys.path.insert(0, "scripts")
import train_cnn_scratch as t, shutil
shutil.copy("outputs/s0/final.pt", "outputs/cnn16s/final_bak.pt")
import os; os.makedirs("outputs/mismatch", exist_ok=True)
shutil.copy("outputs/s0/final.pt", "outputs/mismatch/final.pt"); shutil.copy("outputs/cnn16s/val_predictions.csv", "outputs/mismatch/val_predictions.csv")
t.load_split.__globals__  # noqa
t.predict_test(torch.device("cuda"), "mismatch")
EOF
ls outputs/mismatch
```

**Ran:** Test the ridge test-prediction cell with val standing in

```bash
SP=/tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/376f460f-7079-4666-8d6e-70dc1363b6a4/scratchpad/freeze; mkdir -p $SP/outputs/ridge; cat > $SP/ridge_test_cell.py <<'EOF'
# T15, run once after the freeze (docs/FROZEN_SETTINGS.md): the frozen ridge, fitted on train only, predicts test.
# It must first reproduce the val predictions saved above.
X_tr, y_tr = xy("train", ALL)
final_ridge = ridge().fit(X_tr, np.log(y_tr))
print(f"alpha chosen by leave-one-out on train: {final_ridge[-1].alpha_:.0f}")

saved = pd.read_csv(OUT / "ridge" / "val_predictions.csv")
X_va, _ = xy("val", ALL)
assert np.allclose(saved.pred, np.exp(final_ridge.predict(X_va))), "the refit does not reproduce val_predictions.csv"

X_te, y_te = xy("test", ALL)
pd.DataFrame({"id": splits["test"].id, "age": y_te, "pred": np.exp(final_ridge.predict(X_te))}).to_csv(
    OUT / "ridge" / "test_predictions.csv", index=False)
print(f"{len(y_te)} test predictions saved")
EOF
cd $SP && /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python - <<'EOF'
import json, shutil
from pathlib import Path
REPO = "/home/amaragance/Desktop/Gerfigreynd-Project"
shutil.copy(f"{REPO}/outputs/ridge/val_predictions.csv", "outputs/ridge/val_predictions.csv")
cells = [''.join(c["source"]) for c in json.load(open(f"{REPO}/notebooks/01_baseline.ipynb"))["cells"]]
g = {}
src1 = cells[1].replace('Path("../data")', f'Path("{REPO}/data")').replace('Path("../outputs")', 'Path("outputs")')
exec(src1, g); exec(cells[2], g); exec(cells[3], g)
exec('import pandas as pd\nfeatures = pd.read_csv("' + REPO + '/outputs/features_win30.csv", index_col="id")\nPARTS = {"preN2": (0, 300), "N2": (300, 900)}', g)
exec(cells[5], g)
exec(cells[6].replace('(OUT / "ridge" / "val_predictions.csv"', '(OUT / "ridge" / "val_rerun.csv"'), g)   # keep the copied val file as saved
cell = Path("ridge_test_cell.py").read_text().replace('"test"', '"val"').replace("test_predictions", "standin_predictions")
exec(cell, g)
import pandas as pd, numpy as np
a = pd.read_csv("outputs/ridge/val_predictions.csv"); b = pd.read_csv("outputs/ridge/standin_predictions.csv")
print("stand-in matches saved val predictions:", (a.id == b.id).all(), f"{np.abs(a.pred - b.pred).max():.1e}")
EOF
```

**Ran:** Show the proposed training-script diff

```bash
cat /tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/376f460f-7079-4666-8d6e-70dc1363b6a4/scratchpad/freeze/train_cnn.diff
```

</details>

Everything for the freeze is ready and tested on the real data. There are four parts: the choices to confirm, two code changes, the frozen-settings document, and the run order for tonight.

## Choices to confirm
1. **Both models are trained on train only.** No refit on train plus val; val stays what we used for choosing.
2. **The CNN is trained with three seeds (0, 1, 2), and its headline number is their mean.** Each seed and the range are reported too. The three runs are never averaged into one model, and no seed is picked.
3. **The bootstrap is the same one we used on val.** For each child, the CNN's error is its average absolute error over the three seeds, so the mean over seeds stays consistent with the table.
4. **We only call a difference clear if its 95% interval excludes 0.** Otherwise it's "not distinguishable at this test size".
5. **The val table in the report also switches to the three seed runs.** `cnn16s` becomes the development run and isn't reported.

## Change 1: `scripts/train_cnn.py`
- **`--seed` flag**, so the three runs differ only in their seed.
- **Validation loss in `learning_curve.csv`**, on the same Huber log(age) scale as the training loss. The report's training/validation curve then shares one axis. To get it without a second forward pass, `predict_children` is split into two helpers; its behaviour doesn't change.
- **`--predict-test`**: it reloads `final.pt` and first checks that it reproduces the saved val predictions. If that fails, it stops before predicting test.

```diff
-def predict_children(model, windows, owner, scale, n_children):
-    # every window of every child -> median of the predicted log(age) per child -> years
+def window_log_ages(model, windows, scale):
+    # predicted log(age) of every window
     model.eval()  # always predict in eval mode; train mode normalises each child's amplitude away
     with torch.no_grad():
-        log_pred = torch.cat([model(w.float() / scale) for w in windows.split(512)])
+        return torch.cat([model(w.float() / scale) for w in windows.split(512)])
+
+
+def child_ages(log_pred, owner, n_children):
+    # median of the predicted log(age) over each child's windows -> years
     medians = torch.stack([log_pred[owner == i].quantile(0.5) for i in range(n_children)])
     return medians.exp().cpu().numpy()
 
 
+def predict_children(model, windows, owner, scale, n_children):
+    # every window of every child -> median of the predicted log(age) per child -> years
+    return child_ages(window_log_ages(model, windows, scale), owner, n_children)
+
+
```
In `train()`, replace the `pred = predict_children(...)` line and extend the log and print:
```diff
         recompute_bn(model, X, lengths, scale, crops_per_child, batch_size, seed)
-        pred = predict_children(model, windows, owner, scale, len(val_ids))
+        log_pred = window_log_ages(model, windows, scale)
+        val_loss = nn.functional.huber_loss(log_pred, torch.log(val_ages[owner])).item()  # same loss as training, per window
+        pred = child_ages(log_pred, owner, len(val_ids))
         mae, bal = mae_scores(val_age, pred)
-        log.append({"epoch": epoch, "train_loss": np.mean(losses), "val_MAE": mae,
+        log.append({"epoch": epoch, "train_loss": np.mean(losses), "val_loss": val_loss, "val_MAE": mae,
                     "val_balanced_MAE": bal, "lr": lr_used, "seconds": time.time() - t0})
         pd.DataFrame(log).to_csv(out / "learning_curve.csv", index=False)  # every epoch, in case a run dies
-        print(f"epoch {epoch:3d}  train loss {np.mean(losses):.4f}  val MAE {mae:.2f}  "
+        print(f"epoch {epoch:3d}  train loss {np.mean(losses):.4f}  val loss {val_loss:.4f}  val MAE {mae:.2f}  "
```
Add this new function after `train()`:
```python
def predict_test(device, run_name):
    # T15: run once per frozen model, after the freeze (docs/FROZEN_SETTINGS.md). Nothing is trained or tuned here.
    # The reloaded model must first reproduce the val predictions saved at the end of training.
    run = OUTPUTS / run_name
    ck = torch.load(run / "final.pt", map_location=device, weights_only=True)
    model = BrainAgeCNN(dropout=ck["settings"]["dropout"]).to(device)
    model.load_state_dict(ck["model"])
    for split in ("val", "test"):
        ids, X, lengths, ages = load_split(split, device)
        windows, owner = eval_windows(X, lengths)
        pred = predict_children(model, windows, owner, ck["scale"], len(ids))
        if split == "val":
            saved = pd.read_csv(run / "val_predictions.csv")
            gap = np.abs(saved.pred.to_numpy() - pred).max()
            assert saved.id.tolist() == ids and gap < 1e-3, "final.pt does not reproduce val_predictions.csv"
            print(f"{run_name}: final.pt reproduces val_predictions.csv (largest difference {gap:.1e} years)")
        else:
            pd.DataFrame({"id": ids, "age": ages.cpu().numpy(), "pred": pred}).to_csv(run / "test_predictions.csv", index=False)
            print(f"{run_name}: {len(ids)} test predictions saved to {run / 'test_predictions.csv'}")
```
In `__main__`:
```diff
     parser.add_argument("--run-name", default="cnn16s")
+    parser.add_argument("--seed", type=int, default=0)
+    parser.add_argument("--predict-test", action="store_true",
+                        help="reload --run-name's final.pt and predict test; run once, after the freeze")
 ...
     if args.train:
-        train(device, args.run_name, epochs=args.epochs)
+        train(device, args.run_name, epochs=args.epochs, seed=args.seed)
+    if args.predict_test:
+        predict_test(device, args.run_name)
```

How I tested it, on a scratch copy against your data and outputs:
- **Seed 0, 1 epoch:** val MAE 0.90, against 0.89 at epoch 1 of your `cnn16s` run, within the usual GPU variation.
- **Seed 1:** gives a different run.
- **`predict_test` on your `cnn16s/final.pt`, with val standing in for test so no test data was loaded:** reproduces the saved val predictions to 4.7e-07 years.
- **A mismatched model:** stops at the check.

## Change 2: a new last cell in `notebooks/01_baseline.ipynb`
Add it now so it's part of the freeze commit, but don't run it until step 5 below.
```python
# T15, run once after the freeze (docs/FROZEN_SETTINGS.md): the frozen ridge, fitted on train only, predicts test.
# It must first reproduce the val predictions saved above.
X_tr, y_tr = xy("train", ALL)
final_ridge = ridge().fit(X_tr, np.log(y_tr))
print(f"alpha chosen by leave-one-out on train: {final_ridge[-1].alpha_:.0f}")

saved = pd.read_csv(OUT / "ridge" / "val_predictions.csv")
X_va, _ = xy("val", ALL)
assert np.allclose(saved.pred, np.exp(final_ridge.predict(X_va))), "the refit does not reproduce val_predictions.csv"

X_te, y_te = xy("test", ALL)
pd.DataFrame({"id": splits["test"].id, "age": y_te, "pred": np.exp(final_ridge.predict(X_te))}).to_csv(
    OUT / "ridge" / "test_predictions.csv", index=False)
print(f"{len(y_te)} test predictions saved")
```
I tested it with val standing in for test: alpha is 56 again and the saved val predictions are reproduced. The test features are already in `features_win30.csv`.

## Change 3: `docs/FROZEN_SETTINGS.md`
Written for: the team (Sigga and Amandine), and the instructor if they read the repo.

```markdown
# Frozen settings and test protocol (T14, T15)

Frozen on 16 Nov 2026, at the commit that adds this file (`git log -1 -- docs/FROZEN_SETTINGS.md`).
Everything below was chosen using the training and validation sets only, and none of it changes
after the test predictions are made.

## Data and evaluation
- Data: the instructor's preprocessed recordings (20 bipolar channels, 64 Hz, 0.5–30 Hz) and splits
  (825 / 103 / 104 children). All ages (0.16–16 years) are kept.
- Each recording is used up to its last non-zero sample. 0–300 s is pre-N2, 300–900 s is N2.
- Scores are per child: MAE in years, and balanced MAE, the mean of the MAEs of five age groups
  (<1, 1–3, 3–6, 6–12, 12+, each including its lower edge). r and R² are reported but not used to choose.
- Reference: predicting the training median age (2.98 years) for every child.

## Baseline: ridge regression (notebooks/01_baseline.ipynb)
- 13 features per channel from 30 s windows: log variance, Hjorth mobility and complexity, and log
  and relative power in delta (0.5–4 Hz), theta (4–8), alpha (8–12), sigma (12–16) and beta (16–30),
  from Welch spectra with 4 s segments.
- The median over windows, separately for pre-N2 and N2: 520 features per child.
- Standardised features; ridge regression on log(age); predictions converted back to years. The
  regularisation strength is chosen by leave-one-out on train from 25 values between 0.01 and 10,000
  (chosen: 56). Fitted on train only.

## CNN (scripts/train_cnn.py)
- Architecture: figure 3 of the assignment, with ReLU and batch norm. Three blocks of two (1×3)
  convolutions (16, 32 and 64 filters) and (1×2) max pooling; a fully connected layer per channel with
  64 outputs and batch norm; the maximum over the 20 channels; dropout 0.3 and a linear output for
  log(age). About 549,000 parameters.
- Input: 16 s windows divided by one global scale, the RMS amplitude of the training set (42.4 µV).
- Training on train only: random 16 s crops inside the real signal, 8 per child per epoch, batch 64,
  Huber loss on log(age), AdamW (learning rate 1e-3, weight decay 1e-2), cosine schedule over 60 epochs.
- The model after the last epoch is kept; validation never picks an epoch. Batch-norm statistics are
  recomputed over one fixed epoch of training crops before saving.
- Prediction: every non-overlapping 16 s window of the real signal; the median predicted log(age) per
  child, converted to years.
- Three runs, seeds 0, 1 and 2 (outputs/cnn16s_seed0, _seed1, _seed2). The development run
  outputs/cnn16s is not used in the report.

## Test protocol
1. Train the three CNN runs. Check the comparison notebook on validation before any test
   prediction exists.
2. Predict test once: `--predict-test` for each CNN run and the last cell of 01 for ridge. Both first
   check that the saved model reproduces its validation predictions.
3. CNN numbers are the mean over the three runs; each run's numbers and the range are also reported.
   The runs are not averaged into one model, and no run is picked.
4. Comparison with ridge: the paired bootstrap used on validation (10,000 resamples of children within
   each age group, seed 0). The CNN's error for a child is its mean absolute error over the three runs.
   For MAE and balanced MAE we report the observed difference (CNN − ridge), its 95% interval and the
   share of resamples in which the CNN is worse.
5. A difference is called clear only if its 95% interval excludes 0. Otherwise it is reported as not
   distinguishable at this test size (104 children, 9 aged 12+).
6. The same table, bootstrap and figures are made for validation.
7. If a bug is found after the test predictions are made, we fix it, rerun steps 1–6, and report both
   results and the reason.

Expected before seeing test: test has more children aged 6 or over than validation (30 vs. 23), where
errors are largest, so plain MAE will likely be higher on test for both models.
```

## Run order tonight (fish)
1. Apply the two code changes, then check that the shared code still works:
   ```fish
   python scripts/train_cnn.py --check
   python -u scripts/train_cnn.py --overfit
   ```
   The overfit run should still end around MAE 0.02.
2. Add the document and commit. That commit is the freeze.
3. Close any games, then train the three runs (about 16 min):
   ```fish
   for s in 0 1 2; python -u scripts/train_cnn.py --train --seed $s --run-name cnn16s_seed$s; end
   ```
4. While they train, I'll prepare the updated comparison notebook (three runs, val and test side by side) and check it on val.
5. Only after that:
   ```fish
   for s in 0 1 2; python scripts/train_cnn.py --predict-test --run-name cnn16s_seed$s; end
   ```
   Then run the new last cell of 01.

Two small things to do alongside:
- **AI log:** the new functions, the val-loss lines, the flags, the 01 cell and this document are AI-written, so add them to the file's AI-assisted mark and the AI_USAGE entry.
- **PROJECT_PLAN.md:** it says "average predictions per recording"; it should say median.

---

## Óli, 2026-11-16 20:29

> ok step 3 worked, you wanted to implement an updated comparison notebook before 5.?

**Claude:**

Yes. The comparison code should be written and checked on val before any test prediction exists; that's step 1 of the protocol. First I'll check what the three runs produced and whether the freeze commit is in.

<details><summary>4 commands, 1 other step</summary>

**Ran:** Check freeze commit and the three seed runs

```bash
cd /home/amaragance/Desktop/Gerfigreynd-Project; git log --oneline -4; git status --short; ls docs; grep -n "def predict_test\|--seed\|val_loss\|def window_log_ages\|def child_ages" scripts/train_cnn.py; tail -3 notebooks/01_baseline.ipynb >/dev/null; /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python - <<'EOF'
import json, pandas as pd, torch
nb=json.load(open('notebooks/01_baseline.ipynb'))
last=''.join(nb['cells'][-1]['source']); print('01 last cell starts:', last[:80].replace('\n',' | '), '| outputs:', len(nb['cells'][-1].get('outputs',[])))
for s in (0,1,2):
    r=f'outputs/cnn16s_seed{s}'
    import os; print(r, sorted(os.listdir(r)))
    c=pd.read_csv(f'{r}/learning_curve.csv'); print('  epochs', len(c), 'cols', list(c.columns)); print('  last:', c.iloc[-1][['train_loss','val_loss','val_MAE','val_balanced_MAE']].round(4).to_dict())
    ck=torch.load(f'{r}/final.pt', weights_only=True); print('  settings', ck['settings'], 'scale', round(ck['scale'],2))
EOF
ls outputs/ outputs/ridge
```

**Ran:** Verify freeze commit contents against tested code

```bash
cd /home/amaragance/Desktop/Gerfigreynd-Project; git log -1 --format='%h %ad %s' --date=iso -- docs/FROZEN_SETTINGS.md; git show --stat 4aee1f3 | tail -8; SP=/tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/376f460f-7079-4666-8d6e-70dc1363b6a4/scratchpad/freeze; echo '--- committed vs tested script (ignoring whitespace/comments):'; diff <(grep -v '^\s*#' scripts/train_cnn.py | tr -d ' \n') <(grep -v '^\s*#' $SP/scripts/train_cnn.py | tr -d ' \n') >/dev/null && echo identical code || diff -w <(grep -v '^\s*#' $SP/scripts/train_cnn.py) <(grep -v '^\s*#' scripts/train_cnn.py) | head -60; git diff docs/PROJECT_PLAN.md | head -20
```

**Ran:** Read the changed parts of the committed training script

```bash
ls /tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/376f460f-7079-4666-8d6e-70dc1363b6a4/ 2>&1 | head; uptime -s; cd /home/amaragance/Desktop/Gerfigreynd-Project; sed -n '120,150p;195,275p;405,440p' scripts/train_cnn.py
```

**Ran:** Read the end of predict_test and the frozen-settings doc

```bash
cd /home/amaragance/Desktop/Gerfigreynd-Project; awk 'NR>=275 && NR<=300' scripts/train_cnn.py; echo ...; tail -8 scripts/train_cnn.py; echo; sed -n '1,12p' docs/FROZEN_SETTINGS.md; grep -n "seed\|Protocol\|protocol\|bootstrap\|clear" docs/FROZEN_SETTINGS.md | head -20
```

**Skill:** {"skill": "dataviz"}

</details>

The figures are static report images in two colours that Óli already uses throughout. I'll check that pair with the colour validator rather than eyeballing it.

<details><summary>4 commands, 2 files read</summary>

**Ran:** Validate the two model colours and check static-figure guidance

```bash
cd /tmp/claude-1000/bundled-skills/2.1.325/776d35eafd4f9615c248686bedbeff3e/dataviz; node scripts/validate_palette.js "#2a78d6,#eb6834" --mode light 2>&1 | tail -25; grep -n -i "categorical\|#2a78d6\|#eb6834" references/palette.md | head -12; grep -n -i -A3 "print\|static\|matplotlib" references/marks-and-anatomy.md | head -30
```

**Ran:** Build and run the updated results notebook on val

```bash
SP=/tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/376f460f-7079-4666-8d6e-70dc1363b6a4/scratchpad/nb; mkdir -p $SP && cd $SP && cat > build.py <<'PYEOF'
import json, sys
from pathlib import Path
REPO = "/home/amaragance/Desktop/Gerfigreynd-Project"
SP = Path.cwd()

C1 = r'''import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path("../scripts").resolve()))
from train_cnn import EDGES, GROUPS

DATA, OUT = Path("../data"), Path("../outputs")
FIG, TAB = OUT / "figures", OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

RUNS = {seed: f"cnn16s_seed{seed}" for seed in (0, 1, 2)}   # the frozen CNN, docs/FROZEN_SETTINGS.md
SPLITS = ["val"]   # add "test" only after step 2 of the test protocol

COLOR = {"ridge": "#2a78d6", "CNN": "#eb6834"}   # the same colour for a model in every figure
MARKER = {"ridge": "o", "CNN": "^"}              # and a shape, so the figures survive greyscale printing
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8})

def save(fig, name):
    # sized for a 7-inch-wide page: 300 dpi PNG for Word, PDF for LaTeX
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")

def split_ages(split):
    seg = torch.load(DATA / f"segments_h1k_{split}.pt", weights_only=True)
    return pd.Series({t[4].removesuffix(".edf"): t[3] for t in seg})

def evaluate(age, pred):
    # the same columns as cell 5 of 01_baseline
    err = np.abs(pred - age)
    group = pd.cut(age, EDGES, right=False, labels=GROUPS)
    per_group = pd.Series(err).groupby(group, observed=False).mean()
    return {"MAE": err.mean(), "balanced MAE": per_group.mean(), **per_group.to_dict(),
            "r": np.corrcoef(age, pred)[0, 1], "R2": r2_score(age, pred)}

TRAIN_MEDIAN = split_ages("train").median()   # the reference model's prediction for every child'''

C2 = r'''# learning curves of the three runs: the line is their mean, the band their range
curves = pd.concat([pd.read_csv(OUT / run / "learning_curve.csv") for run in RUNS.values()])
summary = curves.groupby("epoch").agg(["mean", "min", "max"])

fig, (ax_loss, ax_err) = plt.subplots(1, 2, figsize=(7, 2.8))
panels = [(ax_loss, {"train_loss": "training", "val_loss": "validation"}, "Huber loss on log(age)", "Loss"),
          (ax_err, {"val_MAE": "MAE", "val_balanced_MAE": "balanced MAE"}, "years", "Validation error")]
for ax, columns, ylabel, title in panels:
    for (column, label), style in zip(columns.items(), ["-", "--"]):
        s = summary[column]
        ax.plot(s.index, s["mean"], color="black", linestyle=style, linewidth=1.2, label=label)
… (128 more lines)
```

**Read** `/tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/376f460f-7079-4666-8d6e-70dc1363b6a4/scratchpad/nb/figures/cnn_learning_curves.png`

**Read** `/tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/376f460f-7079-4666-8d6e-70dc1363b6a4/scratchpad/nb/figures/cnn_vs_ridge_val.png`

**Ran:** Check the new bootstrap reproduces the old single-run numbers

```bash
cd /tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/376f460f-7079-4666-8d6e-70dc1363b6a4/scratchpad/nb && /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python - <<'EOF' 2>&1 | grep -v Warning
import json, warnings
cells = json.load(open("cells.json"))
REPO = "/home/amaragance/Desktop/Gerfigreynd-Project"
g = {"display": print}
c1 = cells[0].replace('Path("../scripts")', f'Path("{REPO}/scripts")').replace('DATA, OUT = Path("../data"), Path("../outputs")', f'DATA, OUT = Path("{REPO}/data"), Path("{REPO}/outputs")').replace('FIG, TAB = OUT / "figures", OUT / "tables"', 'FIG, TAB = Path("figures"), Path("tables")')
exec(c1, g)
# regression check: the old single run, through the new bootstrap code
g["RUNS"] = {0: "cnn16s"}
exec(cells[2].replace("pred = {split", "pred = {split"), g)
exec(cells[3].replace('table.iloc[2:].mean()', 'table.iloc[2:].mean()').replace('TAB / f"results_{split}.csv"', '"tables/_regress.csv"'), g)
exec(cells[4].replace('TAB / f"bootstrap_{split}.csv"', '"tables/_regress_b.csv"'), g)
print("old 02 notebook: MAE -0.148 [-0.340, +0.039] 5.9%; balanced -0.188 [-0.603, +0.210] 18.7%")
import numpy as np
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    print("r for a constant prediction:", np.corrcoef(np.array([1.,2,3]), np.full(3, 2.98))[0,1], [str(x.message)[:60] for x in w])
EOF
```

**Ran:** Apply fixes and rerun the notebook headless

```bash
cd /tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/376f460f-7079-4666-8d6e-70dc1363b6a4/scratchpad/nb && /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python - <<'EOF'
from pathlib import Path
s = Path("build.py").read_text()
def rep(old, new):
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)
rep('''CNN_COLUMNS = [f"CNN {seed}" for seed in RUNS]

def results_table(df):
    age = df.age.to_numpy()
    rows = {"predict training median": evaluate(age, np.full(len(age), TRAIN_MEDIAN)),
            "ridge": evaluate(age, df.ridge.to_numpy())}
    for column in CNN_COLUMNS:
        rows[f"CNN, seed {column[-1]}"] = evaluate(age, df[column].to_numpy())
    table = pd.DataFrame(rows).T
    # the frozen protocol: the CNN's numbers are the mean over its three runs, not one averaged model
    table.loc["CNN, mean of 3 runs"] = table.iloc[2:].mean()
    return table

tables = {}
for split in SPLITS:
    tables[split] = results_table(pred[split])
    tables[split].to_csv(TAB / f"results_{split}.csv")
    runs = tables[split].iloc[2:5]
    print''', '''CNN_COLUMNS = [f"CNN {seed}" for seed in RUNS]
CNN_ROWS = [f"CNN, seed {seed}" for seed in RUNS]

def results_table(df):
    age = df.age.to_numpy()
    rows = {"predict training median": evaluate(age, np.full(len(age), TRAIN_MEDIAN)),
            "ridge": evaluate(age, df.ridge.to_numpy())}
    rows["predict training median"]["r"] = np.nan   # undefined for a constant prediction
    for column, row in zip(CNN_COLUMNS, CNN_ROWS):
        rows[row] = evaluate(age, df[column].to_numpy())
    table = pd.DataFrame(rows).T
    # the frozen protocol: the CNN's numbers are the mean over its three runs, not one averaged model
    table.loc["CNN, mean of 3 runs"] = table.loc[CNN_ROWS].mean()
    return table

tables = {}
for split in SPLITS:
    tables[split] = results_table(pred[split])
    tables[split].to_csv(TAB / f"results_{split}.csv")
    runs = tables[split].loc[CNN_ROWS]
    print''')
rep('''        draw = [rng.choice(m, size=len(m), replace=True) for m in members]
        d_mae[b] = err_cnn[np.concatenate(draw)].mean() - err_ridge[np.concatenate(draw)].mean()''',
'''        draw = [rng.choice(m, size=len(m), replace=True) for m in members]
        drawn = np.concatenate(draw)
        d_mae[b] = err_cnn[drawn].mean() - err_ridge[drawn].mean()''')
rep('''        rows[metric] = {"observed": observed, "2.5%": low, "97.5%": high,
                        "share CNN worse": (d > 0).mean(), "clear": not low <= 0 <= high}
    boot[split] = pd.DataFrame(rows).T''', '''        rows[metric] = {"observed": observed, "2.5%": low, "97.5%": high, "share CNN worse": (d > 0).mean()}
    boot[split] = pd.DataFrame(rows).T
    boot[split]["clear"] = (boot[split]["97.5%"] < 0) | (boot[split]["2.5%"] > 0)   # the interval excludes 0''')
Path("build.py").write_text(s)
EOF
rm -rf figures tables; PY=/home/amaragance/miniforge3/envs/rei505m-brainage/bin/python; $PY build.py && $PY -m jupyter nbconvert --to notebook --execute 02_scratch.ipynb --output 02_scratch_run.ipynb 2>&1 | grep -v WARNING | tail -1 && $PY - <<'EOF'
import json
… (9 more lines)
```

</details>

All six cells run cleanly with no warnings. I'll print the exact tested cell sources so what I show you is what ran.

<details><summary>1 command</summary>

**Ran:** Print the tested notebook cells

```bash
cd /tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/376f460f-7079-4666-8d6e-70dc1363b6a4/scratchpad/nb && /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python -c "
import json
for i,c in enumerate(json.load(open('cells.json')),1): print(f'##### CELL {i}'); print(c)
"
```

</details>

Yes. Here is the updated `02_cnn_results.ipynb`, to replace its current six cells. I ran it headless on your real val outputs and it works. No test predictions exist yet, so this is the analysis written and checked before test, as the protocol says.

## What changes from the current 02
- **All three seed runs are used.** The CNN row is their mean, each run is listed, and the range is printed.
- **A `SPLITS` switch.** It stays `["val"]` until test predictions exist; then you add `"test"` and rerun everything unchanged.
- **The table gains a "predict training median" row.** Its r is blank (NaN), because r is undefined for a constant prediction; the old 0.00 was rounding noise.
- **The bootstrap uses each child's error averaged over the three runs**, as the frozen document says, and adds a "clear" column (interval excludes 0). It also checks that the files cover exactly that split's children.
  - Run on the old `cnn16s` alone, the new bootstrap code reproduces your earlier numbers exactly: −0.148 [−0.340, +0.039], 5.9%.
- **The figures are report-sized**: 7 in wide, 9 pt text, saved as 300 dpi PNG and PDF.
  - The learning curve shows training and val loss on one scale; the learning-rate panel is dropped.
  - The comparison figure has two panels with a log age axis, so the under-3s are no longer bunched together. CNN points are triangles, so the figure also works in greyscale.
  - Scatter plots show seed 0, which is fixed in advance, not picked.
- **Tables are saved as CSV** to `outputs/tables/`. The figure and table folders are variables in cell 1, so moving them to REPORT_DATA later is a one-line change.

**Val with the three frozen runs:**

| | MAE | balanced MAE |
|---|---|---|
| ridge | 0.75 | 1.08 |
| CNN, mean of 3 runs | 0.60 | 0.88 |

The three runs span MAE 0.59–0.60 and balanced MAE 0.87–0.89. CNN minus ridge:
- MAE −0.152, interval [−0.340, +0.033];
- balanced MAE −0.195, interval [−0.612, +0.200].

Neither difference is clear on val.

## Cell 1: setup
```python
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path("../scripts").resolve()))
from train_cnn import EDGES, GROUPS

DATA, OUT = Path("../data"), Path("../outputs")
FIG, TAB = OUT / "figures", OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

RUNS = {seed: f"cnn16s_seed{seed}" for seed in (0, 1, 2)}   # the frozen CNN, docs/FROZEN_SETTINGS.md
SPLITS = ["val"]   # add "test" only after step 2 of the test protocol

COLOR = {"ridge": "#2a78d6", "CNN": "#eb6834"}   # the same colour for a model in every figure
MARKER = {"ridge": "o", "CNN": "^"}              # and a shape, so the figures survive greyscale printing
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8})

def save(fig, name):
    # sized for a 7-inch-wide page: 300 dpi PNG for Word, PDF for LaTeX
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")

def split_ages(split):
    seg = torch.load(DATA / f"segments_h1k_{split}.pt", weights_only=True)
    return pd.Series({t[4].removesuffix(".edf"): t[3] for t in seg})

def evaluate(age, pred):
    # the same columns as cell 5 of 01_baseline
    err = np.abs(pred - age)
    group = pd.cut(age, EDGES, right=False, labels=GROUPS)
    per_group = pd.Series(err).groupby(group, observed=False).mean()
    return {"MAE": err.mean(), "balanced MAE": per_group.mean(), **per_group.to_dict(),
            "r": np.corrcoef(age, pred)[0, 1], "R2": r2_score(age, pred)}

TRAIN_MEDIAN = split_ages("train").median()   # the reference model's prediction for every child
```

## Cell 2: learning curves
```python
# learning curves of the three runs: the line is their mean, the band their range
curves = pd.concat([pd.read_csv(OUT / run / "learning_curve.csv") for run in RUNS.values()])
summary = curves.groupby("epoch").agg(["mean", "min", "max"])

fig, (ax_loss, ax_err) = plt.subplots(1, 2, figsize=(7, 2.8))
panels = [(ax_loss, {"train_loss": "training", "val_loss": "validation"}, "Huber loss on log(age)", "Loss"),
          (ax_err, {"val_MAE": "MAE", "val_balanced_MAE": "balanced MAE"}, "years", "Validation error")]
for ax, columns, ylabel, title in panels:
    for (column, label), style in zip(columns.items(), ["-", "--"]):
        s = summary[column]
        ax.plot(s.index, s["mean"], color="black", linestyle=style, linewidth=1.2, label=label)
        ax.fill_between(s.index, s["min"], s["max"], color="black", alpha=0.15, linewidth=0)
    ax.set(xlabel="epoch", ylabel=ylabel, title=title, xlim=(1, summary.index.max()))
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
fig.tight_layout()
save(fig, "cnn_learning_curves")
plt.show()
```

## Cell 3: join the predictions
```python
def load_predictions(split):
    # one row per child: id, age, ridge's prediction and each CNN run's, joined on id (never on row order)
    df = pd.read_csv(OUT / "ridge" / f"{split}_predictions.csv", dtype={"id": str}).rename(columns={"pred": "ridge"})
    for seed, run in RUNS.items():
        cnn = pd.read_csv(OUT / run / f"{split}_predictions.csv", dtype={"id": str})
        df = df.merge(cnn.rename(columns={"age": "age_cnn", "pred": f"CNN {seed}"}), on="id", validate="one_to_one")
        # CNN ages were saved as float32, ridge ages as float64, so compare with a tolerance
        assert np.allclose(df.age, df.age_cnn), f"{run} disagrees with ridge on a child's age"
        df = df.drop(columns="age_cnn")
    ages = split_ages(split)
    assert sorted(df.id) == sorted(ages.index), f"the files do not cover exactly the {split} children"
    assert np.allclose(df.age, ages[df.id]), "an age differs from the split file"
    return df

pred = {split: load_predictions(split) for split in SPLITS}
{split: len(df) for split, df in pred.items()}
```

## Cell 4: results tables
```python
CNN_COLUMNS = [f"CNN {seed}" for seed in RUNS]
CNN_ROWS = [f"CNN, seed {seed}" for seed in RUNS]

def results_table(df):
    age = df.age.to_numpy()
    rows = {"predict training median": evaluate(age, np.full(len(age), TRAIN_MEDIAN)),
            "ridge": evaluate(age, df.ridge.to_numpy())}
    rows["predict training median"]["r"] = np.nan   # undefined for a constant prediction
    for column, row in zip(CNN_COLUMNS, CNN_ROWS):
        rows[row] = evaluate(age, df[column].to_numpy())
    table = pd.DataFrame(rows).T
    # the frozen protocol: the CNN's numbers are the mean over its three runs, not one averaged model
    table.loc["CNN, mean of 3 runs"] = table.loc[CNN_ROWS].mean()
    return table

tables = {}
for split in SPLITS:
    tables[split] = results_table(pred[split])
    tables[split].to_csv(TAB / f"results_{split}.csv")
    runs = tables[split].loc[CNN_ROWS]
    print(f"{split}: CNN runs range over MAE {runs['MAE'].min():.2f}-{runs['MAE'].max():.2f}, "
          f"balanced MAE {runs['balanced MAE'].min():.2f}-{runs['balanced MAE'].max():.2f}")
    display(tables[split].round(2))
```

## Cell 5: paired bootstrap
```python
def paired_bootstrap(age, err_cnn, err_ridge, B=10_000, seed=0):
    # children are drawn with replacement within each age group; one draw scores both models (paired)
    group = pd.cut(age, EDGES, right=False, labels=GROUPS).codes
    members = [np.flatnonzero(group == g) for g in range(len(GROUPS))]
    assert all(len(m) > 0 for m in members), "an age group has no children"
    rng = np.random.default_rng(seed)
    d_mae, d_bal = np.empty(B), np.empty(B)
    for b in range(B):
        draw = [rng.choice(m, size=len(m), replace=True) for m in members]
        drawn = np.concatenate(draw)
        d_mae[b] = err_cnn[drawn].mean() - err_ridge[drawn].mean()
        d_bal[b] = np.mean([err_cnn[d].mean() - err_ridge[d].mean() for d in draw])
    return d_mae, d_bal

boot = {}
for split in SPLITS:
    df, table = pred[split], tables[split]
    age = df.age.to_numpy()
    err_cnn = np.abs(df[CNN_COLUMNS].to_numpy() - age[:, None]).mean(axis=1)   # each child's error, averaged over the 3 runs
    err_ridge = np.abs(df.ridge.to_numpy() - age)
    d_mae, d_bal = paired_bootstrap(age, err_cnn, err_ridge)
    rows = {}
    for metric, d in [("MAE", d_mae), ("balanced MAE", d_bal)]:
        observed = table.loc["CNN, mean of 3 runs", metric] - table.loc["ridge", metric]
        low, high = np.percentile(d, [2.5, 97.5])
        rows[metric] = {"observed": observed, "2.5%": low, "97.5%": high, "share CNN worse": (d > 0).mean()}
    boot[split] = pd.DataFrame(rows).T
    boot[split]["clear"] = (boot[split]["97.5%"] < 0) | (boot[split]["2.5%"] > 0)   # the interval excludes 0
    boot[split].to_csv(TAB / f"bootstrap_{split}.csv")
    print(f"{split}: CNN minus ridge, in years (negative = CNN better)")
    display(boot[split].round(3))
```

## Cell 6: comparison figure
```python
def comparison_figure(df):
    # ridge and one CNN run (seed 0, fixed beforehand) against true age; log age axis, since half the children are under 3
    age = df.age.to_numpy()
    fig, (ax_pred, ax_res) = plt.subplots(1, 2, figsize=(7, 3.2))
    for name, column, label in [("ridge", "ridge", "ridge"), ("CNN", "CNN 0", "CNN (seed 0)")]:
        style = dict(s=16, alpha=0.8, color=COLOR[name], marker=MARKER[name], edgecolor="white", linewidth=0.4, label=label)
        ax_pred.scatter(age, df[column], **style)
        ax_res.scatter(age, df[column] - age, **style)
    lims = (0.1, 25)
    ax_pred.plot(lims, lims, "k--", linewidth=0.8, label="perfect prediction")
    ax_pred.set(xscale="log", yscale="log", xlim=lims, ylim=lims, aspect="equal",
                xlabel="true age (years)", ylabel="predicted age (years)", title="Predicted vs. true age")
    ax_res.axhline(0, color="black", linewidth=0.8)
    ax_res.set(xscale="log", xlim=lims, xlabel="true age (years)", ylabel="predicted − true (years)", title="Residuals")
    for ax in (ax_pred, ax_res):
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.grid(alpha=0.3)
    ax_pred.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax_pred.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return fig

for split in SPLITS:
    fig = comparison_figure(pred[split])
    save(fig, f"cnn_vs_ridge_{split}")
    plt.show()
```

## Then, in this order
1. **Paste the cells, run them on val, and commit before step 5.** That commit shows the analysis was fixed before any test prediction existed.
2. **Step 5:**
   ```fish
   for s in 0 1 2; python scripts/train_cnn.py --predict-test --run-name cnn16s_seed$s; end
   ```
   Then run the last cell of `01_baseline.ipynb`. Each run should print "reproduces val_predictions.csv" before saving its test predictions.
3. **Set `SPLITS = ["val", "test"]` and rerun 02 from the top.** Change nothing else.

Two small things:
- `outputs/figures/cnn_learning_curve.png` (singular) is from the old 02 and now out of date; delete it so it can't end up in the report.
- Mark 02 as AI-assisted in its first cell. For the report: validation loss stays below training loss because the training loss is measured with dropout on random crops.

---

## Óli, 2026-11-16 21:06

> okay everything done, are all relevant outputs now in the output folder? and for the report, i believe a docs file with all the info would be best to pull from to format into the report? what do you think?

**Claude:**

I'll check what's in `outputs/` now, whether the test step ran as planned, and what the test numbers are.

<details><summary>4 commands, 1 file read</summary>

**Ran:** List outputs with times and recent commits

```bash
cd /home/amaragance/Desktop/Gerfigreynd-Project; git log --format='%h %ad %s' --date=iso -6; git status --short; echo; find outputs -type f -newer docs/FROZEN_SETTINGS.md -printf '%TY-%Tm-%Td %TH:%TM  %p\n' | sort; echo; find outputs -type f ! -newer docs/FROZEN_SETTINGS.md -printf '%TY-%Tm-%Td %TH:%TM  %p\n' | sort; ls REPORT_DATA 2>&1
```

**Ran:** Read the results notebook outputs including test

```bash
cd /home/amaragance/Desktop/Gerfigreynd-Project; /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python - <<'EOF'
import json
nb=json.load(open('notebooks/02_cnn_results.ipynb'))
print('cells:', len(nb['cells']))
src0=''.join(nb['cells'][0]['source']); print([l for l in src0.splitlines() if 'SPLITS' in l or 'AI' in l or 'RUNS =' in l])
for i,c in enumerate(nb['cells']):
    for o in c.get('outputs',[]):
        if o.get('output_type')=='error': print(i,'ERROR',o['ename'],o['evalue'])
        if 'text' in o: print(f'[{i}]',''.join(o['text']))
        elif 'data' in o and 'text/plain' in o['data'] and 'image/png' not in o['data']: print(f'[{i}]',''.join(o['data']['text/plain']))
nb=json.load(open('notebooks/01_baseline.ipynb'))
for o in nb['cells'][-1].get('outputs',[]):
    if 'text' in o: print('[01 last]',''.join(o['text']))
EOF
```

**Ran:** Compare repo notebook cells with tested version

```bash
cd /home/amaragance/Desktop/Gerfigreynd-Project; /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python - <<'EOF'
import json, difflib
tested = json.load(open("/tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/376f460f-7079-4666-8d6e-70dc1363b6a4/scratchpad/nb/cells.json"))
repo = [''.join(c['source']) for c in json.load(open('notebooks/02_cnn_results.ipynb'))['cells']]
norm = lambda s: [l.rstrip() for l in s.strip().splitlines() if l.strip()]
for i, (a, b) in enumerate(zip(tested, repo), 1):
    d = [l for l in difflib.unified_diff(norm(a), norm(b), lineterm='', n=0) if l[:1] in '+-' and not l.startswith(('---', '+++'))]
    print(f"cell {i}:", "identical" if not d else d)
EOF
echo '--- .gitignore'; cat .gitignore; echo '--- 01 diff vs freeze commit'; git diff --stat notebooks/01_baseline.ipynb; git diff notebooks/01_baseline.ipynb | grep '^[-+]' | grep -v '"execution_count"\|image/png\|^[-+]\s*"[A-Za-z0-9+/=]\{40,\}' | head -30
```

**Ran:** Check test prediction ranges against figure axes

```bash
cd /home/amaragance/Desktop/Gerfigreynd-Project; /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python -c "
import pandas as pd
for r in ['ridge','cnn16s_seed0','cnn16s_seed1','cnn16s_seed2']:
    d=pd.read_csv(f'outputs/{r}/test_predictions.csv'); print(r, len(d), 'pred range', round(d.pred.min(),2), round(d.pred.max(),2), 'age range', d.age.min(), d.age.max())
"
```

**Read** `outputs/figures/cnn_vs_ridge_test.png`

</details>

The test comparison (T15) is done and went exactly as the protocol set out. The test figure is fine: every prediction falls inside the plotted range.

## Test results
| test (104 children) | MAE | balanced MAE |
|---|---|---|
| predict training median | 3.39 | 4.36 |
| ridge | 0.86 | 1.12 |
| CNN, mean of 3 runs | 0.71 (runs 0.70–0.74) | 0.89 (runs 0.87–0.93) |

CNN minus ridge on test:
- **MAE:** −0.149 years, 95% interval [−0.358, +0.057]; the CNN was worse in 7.6% of resamples.
- **Balanced MAE:** −0.231, interval [−0.502, +0.056]; CNN worse in 5.4%.

Three points about these results:
- **Neither difference is "clear" under our rule.** The CNN had lower error on both val and test, by about 0.15 years, but at about 100 children per split the interval still includes 0. Don't pool val and test after the fact; that isn't in the protocol.
- **The prediction we wrote down beforehand held.** Plain MAE rose from val to test for both models (ridge 0.75 → 0.86, CNN 0.60 → 0.71), while balanced MAE barely moved (1.08 → 1.12 and 0.88 → 0.89).
- **Worth checking during interpretation (T16):** on val the CNN was worse than ridge for ages 3–6, but on test it's better (1.04 vs 1.23). One child of about 12 is predicted roughly 6 years too young by both models, which may be something in that recording. Looking into it is fine; changing a model because of it isn't.

## Are all outputs there?
The main comparison is complete:
- val and test predictions for ridge and all three CNN runs;
- the three learning curves;
- `results_` and `bootstrap_` tables for val and test;
- the three figures, as PNG and PDF.

Still missing for the report:
- **The baseline variants table** from 01 (median, C3-P3 only, pre-N2 only, N2 only) exists only as notebook output and isn't saved as a CSV.
- **"Effects of hyperparameter choice" and "experiments that did not work"** (BN fix, N2-only, under-6s, MSE vs Huber) are still scratch results. They need the experiment flags from goal 3 and validation-only reruns.
- **The dataset section** has no figure here, such as an age distribution per split. That may come from Amandine's or Sigga's exploration work.
- **Nothing in `outputs/` is in git**, so Sigga and Amandine can't see any of it.

## A document to write the report from
Yes, I think that's the right approach, with one condition: it should hold the material, not the sentences. If text I write is formatted into the report, it becomes AI-written report text, and captions count too. So:
- **REPORT_DATA/ holds the files:** `figures/` and `tables/`, written directly by the notebooks (in 02, only the `FIG, TAB =` line changes), plus a `README.md` index: each file, what makes it, and which section uses it. Tracking it in git gives the team the files without the checkpoints.
- **`docs/REPORT_MAP.md` holds the ingredients:** for each required section, the points to make as short bullets, the numbers with the CSV each comes from, which figure or table to use, citations to look up, and who writes it. Sigga's and Amandine's sections get their own entries for them to fill in.
- I draft both; you and the team write the prose; AI_USAGE says the map was AI-drafted.

Commit 01, 02 and PROJECT_PLAN now.

Should I go next with the REPORT_DATA changes (the 02 paths and the baseline-variants table in 01) together with the report-map skeleton? The experiment flags would come after, so their results land straight in REPORT_DATA.

---

## Óli, 2026-11-16 21:23

*(attached an image)*

> okay added the baseline table directly to the docs, dataset section will be done later and outputs is no longer gitignored,
>
> lets do a proper experiment on hyperparam config and choices, could you then summarize the experiments that did not work and why here?
>
> after this we can move onto the report_data folder and translating it into the google docs folder

**Claude:**

I'll design the experiments against your current code. First I'll look at what the plan promised to test, your current `train_cnn.py` (you've reformatted it since the freeze), and what un-ignoring `outputs/` now lets into git.

<details><summary>4 commands</summary>

**Ran:** Check plan promises, gitignore and outputs size

```bash
cd /home/amaragance/Desktop/Gerfigreynd-Project; grep -n -i "huber\|mse\|squared\|loss\|window\|segment length\|8 s\|32 s\|learning rate\|dropout\|ablation\|experiment\|hyperparam" docs/PROJECT_PLAN.md | cut -c1-220; echo '--- gitignore'; cat .gitignore | grep -v '^#' | grep -v '^$'; git status --short | head -30; du -sh outputs; du -ah outputs | sort -h | tail -8; git check-ignore -v outputs/cnn16s_seed0/final.pt
```

**Ran:** Read the model class and the start of train()

```bash
cd /home/amaragance/Desktop/Gerfigreynd-Project; sed -n '85,118p;150,200p' scripts/train_cnn.py
```

**Ran:** Copy the training script to scratch and read its main block

```bash
SP=/tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/376f460f-7079-4666-8d6e-70dc1363b6a4/scratchpad/exp; mkdir -p $SP/scripts $SP/outputs && cp /home/amaragance/Desktop/Gerfigreynd-Project/scripts/train_cnn.py $SP/scripts/train_cnn_repo.py && cd $SP && sed -n '200,215p;380,450p' scripts/train_cnn_repo.py; /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python -c "import sklearn; print('sklearn', sklearn.__version__)"
```

**Ran:** Read the end of the training loop

```bash
cd /tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/376f460f-7079-4666-8d6e-70dc1363b6a4/scratchpad/exp; sed -n '213,252p' scripts/train_cnn_repo.py
```

</details>

I've read all the parts that change. Now I'll write the experiment options into the scratch copy, as exact replacements of your current text so the diff applies cleanly.

<details><summary>4 commands</summary>

**Ran:** Apply experiment options to the scratch training script

```bash
cd /tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/376f460f-7079-4666-8d6e-70dc1363b6a4/scratchpad/exp && /home/amaragance/miniforge3/envs/rei505m-brainage/bin/python - <<'EOF'
from pathlib import Path
s = Path("scripts/train_cnn_repo.py").read_text()
def rep(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)

# 1. model: the batch norm after the fully connected layer can be left out
rep('''    def __init__(self, widths=(16, 32, 64), n_features=64, dropout=0.3):''',
    '''    def __init__(self, widths=(16, 32, 64), n_features=64, dropout=0.3, fc_bn=True):''')
rep('''        self.fc = nn.Sequential(nn.Conv2d(n_in, n_features, (1, t_out)), nn.BatchNorm2d(n_features), nn.ReLU()) # this was collapse prone killing all 64 z feature''',
    '''        fc_layers = [nn.Conv2d(n_in, n_features, (1, t_out))]
        if fc_bn:  # without it the fc ReLUs died during training (experiment exp_no_fc_bn)
            fc_layers.append(nn.BatchNorm2d(n_features))
        self.fc = nn.Sequential(*fc_layers, nn.ReLU())''')

# 2. per-child aggregation: exp() only when the model predicts log(age)
rep('''def child_ages(log_pred, owner, n_children):''', '''def child_ages(log_pred, owner, n_children, log_target=True):''')
rep('''    return medians.exp().cpu().numpy()''', '''    return (medians.exp() if log_target else medians).cpu().numpy()   # a years model's output is already in years''')

# 3. helpers for the experiments
rep('''def train(device, run_name, epochs=60, lr=1e-3, weight_decay=1e-2, dropout=0.3,
          crops_per_child=8, batch_size=64, seed=0, overfit=False):
    settings = {"epochs": epochs, "lr": lr, "weight_decay": weight_decay, "dropout": dropout,
                "crops_per_child": crops_per_child, "batch_size": batch_size, "seed": seed,
                "overfit": overfit, "window_s": WIN // FS}''',
'''PARTS = {"all": (0, N_SAMPLES), "preN2": (0, 300 * FS), "N2": (300 * FS, N_SAMPLES)}


def keep_part(X, lengths, part):
    # only one part of the night (0-300 s pre-N2, 300-901 s N2); lengths stay the end of the real signal
    start, end = PARTS[part]
    lengths = lengths.clamp(max=end) - start
    assert (lengths >= WIN).all(), "a recording has less than one window in this part"
    return X[:, :, start:end], lengths


def keep_children(mask, ids, X, lengths, ages):
    idx = mask.nonzero().squeeze(1)
    return [ids[i] for i in idx.tolist()], X[idx], lengths[idx], ages[idx]


def train(device, run_name, epochs=60, lr=1e-3, weight_decay=1e-2, dropout=0.3,
          crops_per_child=8, batch_size=64, seed=0, overfit=False,
          part="all", max_age=None, fc_bn=True, loss="huber", target="log"):
    # the defaults are the frozen settings (docs/FROZEN_SETTINGS.md); the last five are for the experiments only
    settings = {"epochs": epochs, "lr": lr, "weight_decay": weight_decay, "dropout": dropout,
                "crops_per_child": crops_per_child, "batch_size": batch_size, "seed": seed,
                "overfit": overfit, "window_s": WIN // FS, "part": part, "max_age": max_age,
                "fc_bn": fc_bn, "loss": loss, "target": target}
    loss_fn = {"huber": nn.functional.huber_loss, "mse": nn.functional.mse_loss}[loss]
    to_target = torch.log if target == "log" else (lambda a: a)''')

rep('''    ids, X, lengths, ages = load_split("train", device)
    scale = global_scale(X, lengths)''',
'''    ids, X, lengths, ages = load_split("train", device)
    if max_age is not None:
        ids, X, lengths, ages = keep_children(ages < max_age, ids, X, lengths, ages)
    X, lengths = keep_part(X, lengths, part)
… (49 more lines)
```

**Ran:** Run each experiment for one epoch on real data

```bash
cd /tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/376f460f-7079-4666-8d6e-70dc1363b6a4/scratchpad/exp; PY=/home/amaragance/miniforge3/envs/rei505m-brainage/bin/python
echo "frozen seed0 epoch 1:"; sed -n 2p /home/amaragance/Desktop/Gerfigreynd-Project/outputs/cnn16s_seed0/learning_curve.csv
run() { name=$1; shift; out=$($PY -u scripts/train_cnn_scratch.py --train --epochs 1 --seed 0 --run-name $name "$@" 2>&1); echo "$name [$*]: $(echo "$out" | grep '^epoch' | sed 's/  */ /g')"; echo "$out" | grep -i -E "error|traceback" | head -3; }
run default
run no_fc_bn --no-fc-bn
run mse_log --loss mse
run mse_years --loss mse --target years
run n2 --part N2
run preN2 --part preN2
run under6 --max-age 6
run win8 --window 8
run win32 --window 32
run lr3e-4 --lr 3e-4
run lr3e-3 --lr 3e-3
run noreg --dropout 0 --weight-decay 0
$PY - <<'EOF'
import sys, torch, pandas as pd
sys.path.insert(0, "scripts")
import train_cnn_scratch as t
m = t.BrainAgeCNN()
m.load_state_dict(torch.load("/home/amaragance/Desktop/Gerfigreynd-Project/outputs/cnn16s_seed0/final.pt", weights_only=True)["model"])
print("frozen checkpoint loads into the changed model (strict)")
for r in ["under6", "n2", "mse_years", "win32"]:
    d = pd.read_csv(f"outputs/{r}/val_predictions.csv"); ck = torch.load(f"outputs/{r}/final.pt", weights_only=True)
    print(r, len(d), "children, ages", round(d.age.min(), 2), "-", round(d.age.max(), 2), "| settings:", {k: ck["settings"][k] for k in ("window_s", "part", "max_age", "fc_bn", "loss", "target")})
EOF
```

**Ran:** Compare old and new scripts in deterministic mode

```bash
cd /tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/376f460f-7079-4666-8d6e-70dc1363b6a4/scratchpad/exp; PY=/home/amaragance/miniforge3/envs/rei505m-brainage/bin/python
sed -e 's|DATA = ROOT / "data"|DATA = Path("/home/amaragance/Desktop/Gerfigreynd-Project/data")|' -e "s|OUTPUTS = ROOT / \"outputs\"|OUTPUTS = Path(\"$PWD/outputs\")|" scripts/train_cnn_repo.py > scripts/train_cnn_repo_scratch.py
for mod in train_cnn_repo_scratch train_cnn_scratch; do CUBLAS_WORKSPACE_CONFIG=:4096:8 $PY - $mod <<'EOF' 2>&1 | grep -v -i warn | tail -1
import sys, importlib, torch, pandas as pd
sys.path.insert(0, "scripts")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)
t = importlib.import_module(sys.argv[1])
t.train(torch.device("cuda"), f"det_{sys.argv[1]}", epochs=1, seed=0)
c = pd.read_csv(f"outputs/det_{sys.argv[1]}/learning_curve.csv").iloc[0]
p = pd.read_csv(f"outputs/det_{sys.argv[1]}/val_predictions.csv")
print(f"{sys.argv[1]:24s} train_loss {c.train_loss!r} val_loss {c.val_loss!r} val_MAE {c.val_MAE!r} pred_sum {p.pred.sum()!r}")
EOF
done
```

**Ran:** Test the fish experiment loop with echo

```bash
cd /tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/376f460f-7079-4666-8d6e-70dc1363b6a4/scratchpad/exp && mkdir -p outputs/exp_n2_seed1 && touch outputs/exp_n2_seed1/final.pt && cat > loop.fish <<'EOF'
set experiments \
    "no_fc_bn --no-fc-bn" \
    "n2 --part N2" \
    "preN2 --part preN2" \
    "under6 --max-age 6" \
    "mse_years --loss mse --target years" \
    "mse_log --loss mse" \
    "win8 --window 8" \
    "win32 --window 32" \
    "lr3e-4 --lr 3e-4" \
    "lr3e-3 --lr 3e-3" \
    "noreg --dropout 0 --weight-decay 0"
for e in $experiments
    set flags (string split -- " " $e)
    for s in 0 1 2
        set run exp_$flags[1]_seed$s
        if test -f outputs/$run/final.pt
            continue
        end
        echo python -u scripts/train_cnn.py --train --seed $s --run-name $run $flags[2..-1]
    end
end
EOF
fish loop.fish | head -8; echo ...; fish loop.fish | wc -l; fish loop.fish | grep exp_n2
```

</details>

The loop works: it builds the right flags for each run, and a run that already has a `final.pt` is skipped, so you can stop and restart it. Here's the full diff to show you.

<details><summary>1 command</summary>

**Ran:** Show the experiment-options diff

```bash
cat /tmp/claude-1000/-home-amaragance-Desktop-Gerfigreynd-Project/376f460f-7079-4666-8d6e-70dc1363b6a4/scratchpad/exp/train_cnn_exp.diff
```

</details>

The experiment options are ready and tested. Every option defaults to the frozen setting, so your three frozen runs and the test comparison stay as they are.

## Design
- **One change at a time**, starting from the frozen settings, with the same 3 seeds as the frozen model.
- **Validation only.** Each variant is compared with the frozen runs on the same children, using the same paired bootstrap.
- **This is a sensitivity analysis, not model selection.** It comes after the freeze and the test, so no variant is scored on test and none can change the final model. The report's methods section should present it that way.
- **Three analyses need no new runs**; they'll go in the analysis notebook:
  - best epoch vs. last epoch, from the learning curves;
  - how many of the 64 features stay alive, with and without the batch norm;
  - ridge's validation error across its 25 regularisation strengths.

| run name | change | question |
|---|---|---|
| `no_fc_bn` | no batch norm after the fc layer | the fix we made: what happens without it |
| `n2`, `preN2` | one part of the night only | which sleep part carries the age information (mirrors ridge's rows) |
| `under6` | train on under-6s only; scored on the 80 val children under 6 | does specialising on young children help |
| `mse_years` | MSE on age in years | the plain regression default vs. Huber on log(age); target and loss change together |
| `mse_log` | MSE instead of Huber, still on log(age) | the loss alone |
| `win8`, `win32` | 8 s or 32 s windows | the plan called 16 s "a proposal" |
| `lr3e-4`, `lr3e-3` | learning rate ÷3 or ×3 | sensitivity to the step size |
| `noreg` | dropout 0 and weight decay 0 | does regularisation matter with 825 children |

## Change to `scripts/train_cnn.py`
In the model class:
```diff
-    def __init__(self, widths=(16, 32, 64), n_features=64, dropout=0.3):
+    def __init__(self, widths=(16, 32, 64), n_features=64, dropout=0.3, fc_bn=True):
 ...
-        self.fc = nn.Sequential(nn.Conv2d(n_in, n_features, (1, t_out)), nn.BatchNorm2d(n_features), nn.ReLU()) # this was collapse prone killing all 64 z feature
+        fc_layers = [nn.Conv2d(n_in, n_features, (1, t_out))]
+        if fc_bn:  # without it the fc ReLUs died during training (experiment exp_no_fc_bn)
+            fc_layers.append(nn.BatchNorm2d(n_features))
+        self.fc = nn.Sequential(*fc_layers, nn.ReLU())
```
In `child_ages`:
```diff
-def child_ages(log_pred, owner, n_children):
+def child_ages(log_pred, owner, n_children, log_target=True):
 ...
-    return medians.exp().cpu().numpy()
+    return (medians.exp() if log_target else medians).cpu().numpy()   # a years model's output is already in years
```
Directly above `def train(`, add:
```python
PARTS = {"all": (0, N_SAMPLES), "preN2": (0, 300 * FS), "N2": (300 * FS, N_SAMPLES)}


def keep_part(X, lengths, part):
    # only one part of the night (0-300 s pre-N2, 300-901 s N2); lengths stay the end of the real signal
    start, end = PARTS[part]
    lengths = lengths.clamp(max=end) - start
    assert (lengths >= WIN).all(), "a recording has less than one window in this part"
    return X[:, :, start:end], lengths


def keep_children(mask, ids, X, lengths, ages):
    idx = mask.nonzero().squeeze(1)
    return [ids[i] for i in idx.tolist()], X[idx], lengths[idx], ages[idx]
```
In `train()`:
```diff
 def train(device, run_name, epochs=60, lr=1e-3, weight_decay=1e-2, dropout=0.3,
-          crops_per_child=8, batch_size=64, seed=0, overfit=False):
+          crops_per_child=8, batch_size=64, seed=0, overfit=False,
+          part="all", max_age=None, fc_bn=True, loss="huber", target="log"):
+    # the defaults are the frozen settings (docs/FROZEN_SETTINGS.md); the last five are for the experiments only
     settings = {"epochs": epochs, "lr": lr, "weight_decay": weight_decay, "dropout": dropout,
                 "crops_per_child": crops_per_child, "batch_size": batch_size, "seed": seed,
-                "overfit": overfit, "window_s": WIN // FS}
+                "overfit": overfit, "window_s": WIN // FS, "part": part, "max_age": max_age,
+                "fc_bn": fc_bn, "loss": loss, "target": target}
+    loss_fn = {"huber": nn.functional.huber_loss, "mse": nn.functional.mse_loss}[loss]
+    to_target = torch.log if target == "log" else (lambda a: a)
 ...
     ids, X, lengths, ages = load_split("train", device)
+    if max_age is not None:
+        ids, X, lengths, ages = keep_children(ages < max_age, ids, X, lengths, ages)
+    X, lengths = keep_part(X, lengths, part)
     scale = global_scale(X, lengths)
 ...
         val_ids, Xv, lv, val_ages = load_split("val", device)
+        if max_age is not None:
+            val_ids, Xv, lv, val_ages = keep_children(val_ages < max_age, val_ids, Xv, lv, val_ages)
+        Xv, lv = keep_part(Xv, lv, part)
     windows, owner = eval_windows(Xv, lv)
 ...
-    model = BrainAgeCNN(dropout=dropout).to(device)
+    model = BrainAgeCNN(dropout=dropout, fc_bn=fc_bn).to(device)
     with torch.no_grad():
-        model.head[-1].bias.fill_(torch.log(ages).mean())  # start by predicting the average log(age)
+        model.head[-1].bias.fill_(to_target(ages).mean())  # start by predicting the average (log) age
 ...
-            loss = nn.functional.huber_loss(model(x), torch.log(ages[idx]))
+            loss = loss_fn(model(x), to_target(ages[idx]))
 ...
-        val_loss = nn.functional.huber_loss(
-            log_pred, torch.log(val_ages[owner])
-        ).item()
-        pred = child_ages(log_pred, owner, len(val_ids))
+        val_loss = loss_fn(log_pred, to_target(val_ages[owner])).item()
+        pred = child_ages(log_pred, owner, len(val_ids), log_target=(target == "log"))
```
In `__main__`:
```diff
     parser.add_argument("--seed", type=int, default=0)
+    # experiment options (validation only); every default is the frozen setting
+    parser.add_argument("--window", type=int, default=16, help="window length in seconds")
+    parser.add_argument("--lr", type=float, default=1e-3)
+    parser.add_argument("--dropout", type=float, default=0.3)
+    parser.add_argument("--weight-decay", type=float, default=1e-2)
+    parser.add_argument("--part", choices=list(PARTS), default="all", help="use one part of the night only")
+    parser.add_argument("--max-age", type=float, default=None, help="train and score only children younger than this")
+    parser.add_argument("--no-fc-bn", action="store_true", help="leave out the batch norm after the fully connected layer")
+    parser.add_argument("--loss", choices=["huber", "mse"], default="huber")
+    parser.add_argument("--target", choices=["log", "years"], default="log", help="predict log(age) or age in years")
 ...
     args = parser.parse_args()
+    WIN = args.window * FS   # every function reads the window length from this module-level name
 ...
     if args.train:
-        train(device, args.run_name, epochs=args.epochs, seed=args.seed)
+        train(device, args.run_name, epochs=args.epochs, seed=args.seed, lr=args.lr, dropout=args.dropout,
+              weight_decay=args.weight_decay, part=args.part, max_age=args.max_age,
+              fc_bn=not args.no_fc_bn, loss=args.loss, target=args.target)
```

**How I tested it:**
- **The default path is unchanged.** In PyTorch's deterministic mode, your current script and the changed one give bit-identical training loss, val loss and predictions for 1 epoch.
- **All 11 experiments ran for 1 epoch** on the real data without errors, and the saved settings record each change. The `under6` run scores exactly the 80 val children under 6.
- **Your frozen checkpoints still load** into the changed model.

## Running it (about 3.2 hours)
Close any games, run `conda activate rei505m-brainage`, then paste this into fish from the repo root:
```fish
set experiments \
    "no_fc_bn --no-fc-bn" \
    "n2 --part N2" \
    "preN2 --part preN2" \
    "under6 --max-age 6" \
    "mse_years --loss mse --target years" \
    "mse_log --loss mse" \
    "win8 --window 8" \
    "win32 --window 32" \
    "lr3e-4 --lr 3e-4" \
    "lr3e-3 --lr 3e-3" \
    "noreg --dropout 0 --weight-decay 0"
for e in $experiments
    set flags (string split -- " " $e)
    for s in 0 1 2
        set run exp_$flags[1]_seed$s
        if test -f outputs/$run/final.pt   # already done, so the loop can be stopped and restarted
            continue
        end
        python -u scripts/train_cnn.py --train --seed $s --run-name $run $flags[2..-1] 2>&1 | tee -a outputs/experiments.log
    end
end
```
- **It's 33 runs.** The five "did not work" candidates come first and finish in about 80 minutes; the hyperparameter runs follow.
- **You can stop it** with Ctrl-C and paste it again later; finished runs are skipped.
- **Only small files land in git.** Each run writes `outputs/exp_<name>_seed<s>/`; the checkpoints stay out through `*.pt`.

When it's done, I'll build `03_experiments.ipynb`: one table and one figure of each variant's difference from the frozen model with its 95% interval, plus the three no-run analyses. Then I'll summarise here which experiments didn't work and why.

Add the new functions, flags and the experiment loop to the AI-assisted mark and the AI log.
