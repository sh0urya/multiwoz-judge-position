"""Prepare and run the first turn-position experiment."""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
from urllib.request import Request, urlopen
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent
ARCHIVE_HASH = "d4c66523614af016c1d2dafa35b4a548c96cb9ffadb71338937b60ac51272fe5"
POSITIONS = ("early", "25%", "middle", "75%", "late")
J1_PROMPT = ROOT / "prompts" / "j1.txt"
J3_PROMPT = ROOT / "prompts" / "j3.txt"


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(args):
    if digest(args.archive) != ARCHIVE_HASH:
        raise ValueError("The dataset archive does not match the verified MultiWOZ 2.4 release.")

    with ZipFile(args.archive) as archive:
        dialogues = json.loads(archive.read("MULTIWOZ2.4/data.json"))
        test_ids = archive.read("MULTIWOZ2.4/testListFile.json").decode().split()

    candidates = {}
    for dialogue_id in sorted(test_ids):
        messages = dialogues[dialogue_id]["log"]
        if len(messages) < 20 or len(messages) % 2:
            continue
        candidates[dialogue_id] = [message["text"] for message in messages]

    edits = json.loads(args.edits.read_text(encoding="utf-8"))
    pilot_ids = [edit["dialogue_id"] for edit in edits]
    if len(pilot_ids) != len(set(pilot_ids)) or not set(pilot_ids) <= candidates.keys():
        raise ValueError("Each pilot dialogue must be unique and eligible.")

    # Reserve the development examples so their results cannot leak into the final sample.
    remaining = sorted(candidates.keys() - set(pilot_ids))
    candidate_ids = random.Random(args.seed).sample(remaining, args.count)
    candidate_records = [
        {"dialogue_id": dialogue_id, "messages": candidates[dialogue_id]}
        for dialogue_id in candidate_ids
    ]

    cases = []
    for edit in edits:
        dialogue_id = edit["dialogue_id"]
        messages = candidates[dialogue_id]
        cases.append({"dialogue_id": dialogue_id, "position": "clean", "messages": messages})
        last_index = len(messages) // 2 - 1
        for quartile, position in enumerate(POSITIONS):
            # Map each quartile to a zero-based assistant turn, rounding ties upward.
            turn_index = (quartile * last_index + 2) // 4
            message_index = 2 * turn_index + 1
            perturbed = messages.copy()
            perturbed[message_index] = f"{messages[message_index]} {edit['false_claim']}"
            cases.append(
                {
                    **edit,
                    "position": position,
                    "expected_turn": turn_index + 1,
                    "messages": perturbed,
                }
            )

    write_jsonl(args.candidates, candidate_records)
    write_jsonl(args.pilot, cases)
    print(
        f"{len(candidates)} eligible test dialogues. "
        f"Wrote {len(candidate_records)} candidates and {len(cases)} pilot cases."
    )


def transcript(messages):
    lines = []
    for index, message in enumerate(messages):
        speaker = "User" if index % 2 == 0 else "Assistant"
        lines.append(f"{speaker} {index // 2 + 1}: {message}")
    return "\n".join(lines)


def parse_judgment(content, turns, focus_turn=None):
    content = content.strip()
    if content.startswith("```") and content.endswith("```"):
        content = content.split("\n", 1)[1][:-3].strip()
    result = json.loads(content)
    if not isinstance(result, dict):
        raise ValueError("Expected a judgment object.")
    score = result.get("score")
    if type(score) is not int or not 1 <= score <= 10:
        raise ValueError("Expected an integer score from 1 to 10.")
    if focus_turn is not None:
        problem = result.get("problem")
        if not isinstance(problem, str):
            raise ValueError("Expected a problem description.")
        issues = [{"turn": focus_turn, "problem": problem.strip()}] if problem.strip() else []
        return {"score": score, "issues": issues}
    issues = result.get("issues")
    if not isinstance(issues, list):
        raise ValueError("Expected a list of issues.")
    if any(
        not isinstance(issue, dict)
        or type(issue.get("turn")) is not int
        or not 1 <= issue["turn"] <= turns
        or not isinstance(issue.get("problem"), str)
        or not issue["problem"].strip()
        for issue in issues
    ):
        raise ValueError("The judge returned an issue without a valid assistant turn and explanation.")
    return result


def judgment_format(turnwise):
    score = {"type": "integer", "minimum": 1, "maximum": 10}
    problem = {"type": "string", "maxLength": 300}
    if turnwise:
        fields = {"score": score, "problem": problem}
    else:
        issue = {
            "type": "object",
            "properties": {"turn": {"type": "integer", "minimum": 1}, "problem": problem},
            "required": ["turn", "problem"],
            "additionalProperties": False,
        }
        fields = {"score": score, "issues": {"type": "array", "items": issue, "maxItems": 3}}
    return {
        "type": "json_object",
        "schema": {
            "type": "object",
            "properties": fields,
            "required": list(fields),
            "additionalProperties": False,
        },
    }


def judge(args):
    turnwise = args.strategy == "j3"
    prompt_path = args.prompt or (J3_PROMPT if turnwise else J1_PROMPT)
    prompt = prompt_path.read_text(encoding="utf-8")
    cases = read_jsonl(args.pilot)
    random.Random(args.seed).shuffle(cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Content-Type": "application/json"}
    if api_key := os.environ.get("JUDGE_API_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"
    run_settings = {
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "prompt_sha256": digest(prompt_path),
        "cases_sha256": digest(args.pilot),
    }
    if turnwise:
        run_settings["strategy"] = args.strategy
    response_format = judgment_format(turnwise) if args.structured else {"type": "json_object"}
    if args.structured:
        run_settings["response_format"] = response_format
    existing = read_jsonl(args.output) if args.output.exists() else []
    if any(record["settings"] != run_settings for record in existing):
        raise ValueError("The output already contains judgments from different run settings.")
    completed = {
        (record["dialogue_id"], record["position"], record.get("focus_turn"))
        for record in existing
        if record["judgment"] is not None
    }
    reuse_prefixes = turnwise and args.temperature == 0
    prefix_cache = {}
    if reuse_prefixes:
        messages_by_case = {
            (case["dialogue_id"], case["position"]): case["messages"] for case in cases
        }
        for record in existing:
            if record["judgment"] is not None:
                case_id = (record["dialogue_id"], record["position"])
                messages = messages_by_case[case_id][: 2 * record["focus_turn"]]
                prefix_cache.setdefault(tuple(messages), record)

    for case in cases:
        case_id = (case["dialogue_id"], case["position"])
        turns = len(case["messages"]) // 2
        focus_turns = range(1, turns + 1) if turnwise else [None]
        for focus_turn in focus_turns:
            if (*case_id, focus_turn) in completed:
                continue
            messages = case["messages"][: 2 * focus_turn] if focus_turn else case["messages"]
            reused = prefix_cache.get(tuple(messages))
            if reused:
                completion = reused["response"]
            else:
                payload = {
                    "model": args.model,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "seed": args.seed,
                    "response_format": response_format,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": transcript(messages)},
                    ],
                }
                request = Request(
                    f"{args.base_url.rstrip('/')}/chat/completions",
                    data=json.dumps(payload).encode(),
                    headers=headers,
                )
                with urlopen(request, timeout=args.timeout) as response:
                    completion = json.load(response)
            choice = completion["choices"][0]
            content = choice["message"].get("content") or ""
            validation_error = None
            try:
                judgment = parse_judgment(content, turns, focus_turn)
            except (ValueError, IndexError) as exc:
                judgment = None
                validation_error = str(exc)
            record = {
                "dialogue_id": case["dialogue_id"],
                "position": case["position"],
                "settings": run_settings,
                "judgment": judgment,
                "validation_error": validation_error,
                "response": completion,
            }
            if focus_turn is not None:
                record["focus_turn"] = focus_turn
            if reused:
                record["reused_from"] = [reused["dialogue_id"], reused["position"]]
            elif reuse_prefixes and judgment is not None:
                prefix_cache[tuple(messages)] = record
            with args.output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            status = judgment["score"] if judgment else "invalid"
            label = f" turn {focus_turn}" if focus_turn else ""
            print(f"{case_id[0]} {case_id[1]}{label}: {status}", flush=True)


def dialogue_judgments(cases, records):
    judgments = defaultdict(dict)
    for record in records:
        if record["judgment"] is not None:
            case_id = (record["dialogue_id"], record["position"])
            judgments[case_id][record.get("focus_turn")] = record["judgment"]
    results = {}
    for case in cases:
        case_id = (case["dialogue_id"], case["position"])
        by_turn = judgments.get(case_id, {})
        if None in by_turn:
            results[case_id] = by_turn[None]
        elif len(by_turn) == len(case["messages"]) // 2:
            results[case_id] = {
                "score": min(item["score"] for item in by_turn.values()),
                "issues": [issue for item in by_turn.values() for issue in item["issues"]],
            }
    return results


def summarize(args):
    cases = read_jsonl(args.pilot)
    results = dialogue_judgments(cases, read_jsonl(args.output))
    reviews = json.loads(args.reviews.read_text(encoding="utf-8")) if args.reviews else {}
    grouped = defaultdict(list)
    for case in cases:
        dialogue_id, position = case["dialogue_id"], case["position"]
        result = results.get((dialogue_id, position))
        clean = results.get((dialogue_id, "clean"))
        if position != "clean" and result and clean:
            reviewed = reviews.get(dialogue_id, {}).get(position)
            grouped[position].append((clean["score"] - result["score"], reviewed))
    clean_count = sum(position == "clean" for _, position in results)
    print(f"Scored clean controls: {clean_count}")
    print("position  pairs   mean score drop (points)   injected errors detected")
    for position in POSITIONS:
        if not grouped[position]:
            print(f"{position:<9} 0")
            continue
        drops, detections = zip(*grouped[position])
        reviewed = [detected for detected in detections if type(detected) is bool]
        label = f"{sum(reviewed)}/{len(reviewed)} reviewed" if reviewed else "unreviewed"
        print(f"{position:<9} {len(drops):>5}   {statistics.mean(drops):>24.2f}   {label}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)

    prep = commands.add_parser("prepare", help="Build a candidate sample and matched pilot.")
    prep.add_argument("--archive", type=Path, default=ROOT / "data" / "MULTIWOZ2.4.zip")
    prep.add_argument("--edits", type=Path, default=ROOT / "data" / "pilot_edits.json")
    prep.add_argument("--candidates", type=Path, default=ROOT / "data" / "candidates.jsonl")
    prep.add_argument("--pilot", type=Path, default=ROOT / "data" / "pilot.jsonl")
    prep.add_argument("--count", type=int, default=40)
    prep.add_argument("--seed", type=int, default=42)
    prep.set_defaults(run=prepare)

    run = commands.add_parser("judge", help="Run blind judgments through a chat endpoint.")
    run.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    run.add_argument("--strategy", choices=("j1", "j3"), default="j1")
    run.add_argument("--prompt", type=Path, help="Override the default prompt for the chosen strategy.")
    run.add_argument("--structured", action="store_true", help="Constrain JSON using the llama.cpp schema format.")
    run.add_argument("--model", required=True)
    run.add_argument("--pilot", type=Path, default=ROOT / "data" / "pilot.jsonl")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--max-tokens", type=int, default=400)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--timeout", type=int, default=120)
    run.set_defaults(run=judge)

    summary = commands.add_parser("summarize", help="Compare each edited score with its clean control.")
    summary.add_argument("--pilot", type=Path, default=ROOT / "data" / "pilot.jsonl")
    summary.add_argument("--output", type=Path, required=True)
    summary.add_argument("--reviews", type=Path)
    summary.set_defaults(run=summarize)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
