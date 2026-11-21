"""Summarize the exploratory runs without modifying any judgments."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean

from experiment import read_jsonl


def normalized(value):
    return " ".join(value.casefold().strip().strip(" .!?\"'").split())


def summarized_stage(manifest, output, strategy):
    cases = read_jsonl(manifest)
    cases_by_id = {case["case_id"]: case for case in cases}
    outputs = {}
    if output.exists():
        for row in read_jsonl(output):
            if row["case_id"] not in cases_by_id:
                raise ValueError(f"Result has unknown case ID {row['case_id']}")
            key = (row["case_id"], row["focus_turn"])
            if row["judgment"] is not None or key not in outputs:
                outputs[key] = row

    counts = Counter()
    by_group = defaultdict(Counter)
    pairs = defaultdict(dict)
    drops = defaultdict(list)
    expected_records = 0
    for case in cases:
        turns = range(1, len(case["messages"]) // 2 + 1) if strategy == "j3" else [None]
        expected_records += len(turns)
        rows = [outputs.get((case["case_id"], turn)) for turn in turns]
        if any(row is None or row["judgment"] is None for row in rows):
            continue
        judgments = [row["judgment"] for row in rows]
        target = case["expected_turn"]
        flags = [
            issue
            for judgment in judgments
            for issue in judgment["issues"]
            if issue["turn"] == target
        ]
        group = by_group[(case["family"], case["position"])]
        condition = case["condition"]
        group[condition + "_cases"] += 1
        counts[condition + "_cases"] += 1
        if condition == "incorrect":
            strict_hit = any(
                issue.get("supporting_user_turn") == case["evidence_turn"]
                and normalized(issue.get("observed_value") or "") == normalized(case["incorrect_value"])
                and normalized(issue.get("expected_value") or "") == normalized(case["correct_value"])
                for issue in flags
            )
            if strict_hit:
                group["evidence_matched_hits"] += 1
                counts["evidence_matched_hits"] += 1
            elif flags:
                group["other_target_flags_on_errors"] += 1
                counts["other_target_flags_on_errors"] += 1
        elif flags:
            group["target_flags_on_controls"] += 1
            counts["target_flags_on_controls"] += 1

        pairs[case["pair_id"]][condition] = {
            "dialogue_score": min(judgment["score"] for judgment in judgments),
            "position": case["position"],
            "target_score": judgments[target - 1]["score"] if strategy == "j3" else None,
        }

    for pair in pairs.values():
        if "correct" not in pair or "incorrect" not in pair:
            continue
        correct, incorrect = pair["correct"], pair["incorrect"]
        change = correct["dialogue_score"] - incorrect["dialogue_score"]
        drops["all_positions"].append(change)
        drops[correct["position"]].append(change)
        if strategy == "j3":
            drops["target_reply_all_positions"].append(correct["target_score"] - incorrect["target_score"])

    return {
        "strategy": strategy,
        "expected_cases": len(cases),
        "expected_records": expected_records,
        "valid_records": sum(row["judgment"] is not None for row in outputs.values()),
        "invalid_records": sum(row["judgment"] is None for row in outputs.values()),
        "reused_records": sum("reused_from" in row and row["judgment"] is not None for row in outputs.values()),
        "totals": dict(counts),
        "by_family_and_position": [
            {"family": family, "position": position, **metrics}
            for (family, position), metrics in sorted(by_group.items())
        ],
        "paired_score_drops": {
            position: {
                "pairs": len(values),
                "mean": round(mean(values), 4),
                "decreases": sum(value > 0 for value in values),
                "ties": sum(value == 0 for value in values),
                "increases": sum(value < 0 for value in values),
            }
            for position, values in drops.items()
        },
        "labeling": (
            "Automated exact-value and evidence-turn matching. Target flags on controls "
            "and other target flags on edits require independent review."
        ),
    }


def write_summary(path, summary):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategy", choices=("j1", "j3"), required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    summary = summarized_stage(args.manifest, args.output, args.strategy)
    if args.summary:
        write_summary(args.summary, summary)
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
