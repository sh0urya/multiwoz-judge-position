# Dialogue judge position pilot

- [Updated milestone PDF](milestone/milestone.pdf)
- [Qwen experiment brief](milestone/qwen-experiment-brief.pdf)
- [Blinded pilot review packet](overnight/pilot/review/transcripts.md)
- Full-batch summaries for [J1](overnight/full/j1-summary.json) and [J3](overnight/full/j3-summary.json)
- [Original milestone backup](milestone/milestone%20backup.pdf)

The overnight results are preliminary pending team review. For the blind pass, start with the review packet and record your judgments before opening the answer keys or model outputs.

The pilot evaluates the same train-route contradiction at five positions in each of three MultiWOZ 2.4 dialogues. Each dialogue has an original control. The script also selects 40 separate candidates from the official test split with seed 42. The candidate list still needs review before it becomes the final dataset.

## Prepare the pilot

Use Python 3.11 or later. The experiment uses only the Python standard library.

```sh
mkdir -p data
curl -fL https://raw.githubusercontent.com/smartyfh/MultiWOZ2.4/main/data/MULTIWOZ2.4.zip -o data/MULTIWOZ2.4.zip
python3 experiment.py prepare
```

The [official dataset archive](https://github.com/smartyfh/MultiWOZ2.4/tree/main/data) is included with the [upstream license](data/MultiWOZ-LICENSE.txt). The script verifies the archive against its SHA-256 hash. It interprets a turn as a user message and the subsequent assistant reply, keeps all dialogue text verbatim, and appends one sentence to the target reply in each edited copy. The same sentence is used across five positions within each dialogue. The five positions are the first reply, nearest replies to the first, second, and third quartiles of the reply indices, and final reply. The original replies at these positions can serve different purposes, and the appended sentence may be conspicuous in closing replies.

## Run a judge

The pilot uses [llama.cpp b12248](https://github.com/ggml-org/llama.cpp/releases/tag/b12248) and [Qwen3.5-9B Q4_K_M weights from Unsloth](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/blob/main/Qwen3.5-9B-Q4_K_M.gguf). The verified model SHA-256 is `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8`. Start a local server with the downloaded model.

```sh
llama-server -m /path/to/Qwen3.5-9B-Q4_K_M.gguf \
  --host 127.0.0.1 --port 18129 -c 4096 -np 1 --reasoning off \
  --alias Qwen3.5-9B-Q4_K_M
```

From a second terminal, working in this directory:

```sh
python3 experiment.py judge --structured --strategy j1 --base-url http://127.0.0.1:18129/v1 \
  --model Qwen3.5-9B-Q4_K_M --output results/j1-qwen-9b.jsonl
python3 experiment.py judge --structured --strategy j3 --base-url http://127.0.0.1:18129/v1 \
  --model Qwen3.5-9B-Q4_K_M --output results/j3-qwen-9b.jsonl
```

J1 grades the whole transcript once. J3 adapts the [MT-Bench-101](https://aclanthology.org/2024.acl-long.401/) pattern to this domain. It grades each assistant reply with the preceding user and assistant messages as context and uses the minimum turn grade as the dialogue grade. Both use a 1–10 scale, temperature 0, and seed 42. The judge sees only the rubric and conversation text. At temperature zero, J3 reuses responses for identical transcript prefixes and marks these copies with `reused_from` in the logs. Run results retain the raw responses and can resume using the same command. Use a separate output path for each run configuration.

The initial 0.8B format check used `prompts/j1_format_example.txt` and returned 9 with no issues on all 18 cases. The main prompt describes the format without a numeric example. Run on the [0.8B GGUF](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/blob/main/Qwen3.5-0.8B-Q4_K_M.gguf) with the same server options, port 18128, and alias `Qwen3.5-0.8B-Q4_K_M` to reproduce the smaller-model J1 pilot.

## Review and summarize

Each review entry is keyed by dialogue ID and position. Count a detection if an issue at the edited turn explicitly identifies the incorrect route claim. An incorrect explanation of the correct route is recorded as a qualification, while merely repeating the claim or criticizing a different detail counts as a miss. The reviews are first-pass labels to be independently verified by the team. In the 0.8B run, PMUL0509's late reply was flagged for using Ely as the departure city, but the rationale confused the correct origin with the destination.

```sh
python3 experiment.py summarize --output results/j1-qwen-9b.jsonl \
  --reviews results/review-j1-qwen-9b.json
python3 experiment.py summarize --output results/j3-qwen-9b.jsonl \
  --reviews results/review-j3-qwen-9b.json
```

The summary reports score drops in original 1–10 score points. Divide by nine to convert score differences to the normalized scale, retaining negative values when an edited copy gets a higher score.

## Run the correct-route controls

`python3 route_controls.py` pairs each of the 15 false-route edits with a version containing the correct route value. It keeps the appended sentence at the same reply and changes only the place name. The correct-route counterparts use a separate input and output file because they share the dialogue IDs and positions of the original experiment.

```sh
python3 route_controls.py
python3 experiment.py judge --structured --strategy j1 --base-url http://127.0.0.1:18129/v1 \
  --model Qwen3.5-9B-Q4_K_M --pilot data/route_controls.jsonl \
  --output results/j1-qwen-9b-route-controls.jsonl
```

The annotations in `results/review-j1-qwen-9b-route-controls.json` record whether J1 treated the correct route as wrong or unsupported, and whether it objected to the appended sentence's timing, repetition, or relevance. The review uses the text of the reported issue and its cited turn. It is targeted to these two outcomes and leaves other reported quality issues in the raw logs. The late Peterborough output objects to the route and then acknowledges it is the requested destination, so the review marks its rationale as ambiguous. The Norwich middle output is tagged separately because it misreads the train city as a restaurant location.

## Rebuild the milestone draft

Run `python3 plot_pilot.py` with Matplotlib 3.11.0 to regenerate the position curve from the two reviewed 9B runs. In the `milestone` directory, run `pdflatex milestone-overnight.tex`, `bibtex milestone-overnight`, and `pdflatex milestone-overnight.tex` twice to build the updated ACL-style PDF as `milestone-overnight.pdf`. The initial draft source is `milestone.tex`.

The pilot was prepared with Codex assistance, including experiment code, exploratory review labels, and milestone drafting. The team should review the code, labels, and draft before course submission.

## Overnight exploratory batch

This batch uses constructed train readback exchanges grounded in bundled MultiWOZ test-set goals. The six pilot sources and 40 full-batch sources are disjoint. The 36-transcript pilot covers destination and day at three positions. The full manifest crosses departure, destination, day, time cutoff, and party size with five positions. Correct and incorrect members of each pair differ in one field in the readback. The supporting request stays inside that exchange as it moves between the same filler exchanges. The provenance logs identify the source fields and explicitly record constructed party sizes.

The full batch has 1,000 edit/control pairs and 1,200 distinct whole-transcript inputs. Five correct transcripts per source are shared across error families. J3 also reuses identical prefixes. The run preserves case IDs in both situations and records which raw judgment the reused result links to.

Prepare the manifests in a new directory to reproduce the data without touching an existing run.

```sh
python3 overnight_dataset.py --out-dir overnight-reproduction
```

Launch or resume the local run from this directory. Use the same directory and inputs to resume existing checkpoints.

```sh
/usr/bin/caffeinate -i -s python3 run_overnight.py \
  --server /path/to/llama-server \
  --weights /path/to/Qwen3.5-9B-Q4_K_M.gguf --workers 8
```

The sequence is pilot J1, pilot J3, full J1, then full J3. The server binds to loopback, with thinking disabled. Stage logs and summaries are in `overnight/pilot` and `overnight/full`. Use `overnight/status.json` for the active stage and `review/transcripts.md` inside the pilot directory for independent review. The latter contains blinded transcripts. The summaries use exact conflicting-value and evidence-turn matching as a conservative automated count. Human review should adjudicate target flags on correct controls and other target flags on edited cases before the results support conclusions about position.
