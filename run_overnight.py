"""Run the pilot and exploratory Qwen batch in order on this Mac."""

import argparse
from collections import defaultdict
from datetime import datetime
from fcntl import LOCK_EX, LOCK_NB, flock
from http.client import HTTPConnection, HTTPException
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from experiment import ROOT, digest, read_jsonl
from overnight_report import summarized_stage, write_summary


MODEL = "Qwen3.5-9B-Q4_K_M"
MODEL_SHA256 = "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8"
STAGES = (("pilot", "j1"), ("pilot", "j3"), ("full", "j1"), ("full", "j3"))


def checked_manifest(path):
    cases = read_jsonl(path)
    by_id = {case["case_id"]: case for case in cases}
    if not cases or len(by_id) != len(cases):
        raise ValueError(f"Missing cases or duplicate case IDs in {path}")
    pairs = defaultdict(dict)
    for case in cases:
        condition = case["condition"]
        pair = pairs[case["pair_id"]]
        if condition not in {"correct", "incorrect"} or condition in pair:
            raise ValueError(f"Invalid pair membership in {case['case_id']}")
        pair[condition] = case

    shared = ("dialogue_id", "family", "position", "expected_turn", "evidence_turn",
              "correct_value", "incorrect_value")
    for pair_id, pair in pairs.items():
        if set(pair) != {"correct", "incorrect"}:
            raise ValueError(f"Missing counterpart for {pair_id}")
        good, bad = pair["correct"], pair["incorrect"]
        if any(good[field] != bad[field] for field in shared):
            raise ValueError(f"Pair metadata differs in {pair_id}")
        target = 2 * good["expected_turn"] - 1
        evidence = 2 * good["evidence_turn"] - 2
        messages = good["messages"].copy()
        correct, incorrect = good["correct_value"], good["incorrect_value"]
        if (
            len(messages) < 20
            or len(messages) % 2
            or not 0 <= evidence < target < len(messages)
            or not correct
            or not incorrect
            or correct == incorrect
            or correct not in messages[evidence]
        ):
            raise ValueError(f"Missing or ambiguous evidence in {pair_id}")
        before, after = messages[target], bad["messages"][target]
        replacements = sum(
            before[:match.start()] + incorrect + before[match.end():] == after
            for match in re.finditer(re.escape(correct), before)
        )
        messages[target] = after
        if replacements != 1 or messages != bad["messages"]:
            raise ValueError(f"Additional edit in {pair_id}")
    return cases


def wait_for_model(server, port):
    expires = time.monotonic() + 120
    while time.monotonic() < expires:
        if server.poll() is not None:
            raise RuntimeError("The local model server exited. Check server.log.")
        connection = HTTPConnection("127.0.0.1", port, timeout=1)
        try:
            connection.request("GET", "/health")
            response = connection.getresponse()
            response.read()
            if response.status == 200:
                return
        except (OSError, HTTPException):
            pass
        finally:
            connection.close()
        time.sleep(1)
    raise TimeoutError("The local model server did not become ready. Check server.log.")


def run_batch(args):
    work = args.directory
    status_path = work / "status.json"
    milestone = ROOT / "milestone" / "milestone.pdf"
    original_hash = digest(milestone)
    manifests = {name: work / name / "cases.jsonl" for name in ("pilot", "full")}
    pilot = checked_manifest(manifests["pilot"])
    full = checked_manifest(manifests["full"])
    if len(pilot) != 36 or len(full) > 2000:
        raise ValueError("Expected 36 pilot cases and at most 2,000 full-batch cases.")
    pilot_sources = {case["dialogue_id"] for case in pilot}
    if pilot_sources & {case["dialogue_id"] for case in full}:
        raise ValueError("Pilot and full batch share a source dialogue.")
    if digest(args.weights) != MODEL_SHA256:
        raise ValueError("The local model weights do not match the verified Qwen file.")

    status = {
        "active_stage": None,
        "state": "starting",
        "completed_stages": [],
        "pilot_cases": len(pilot),
        "full_edited_cases": len(full) // 2,
        "full_control_cases": len(full) // 2,
        "model": MODEL,
        "milestone_original_sha256": original_hash,
    }

    def update(**fields):
        status.update(fields, updated_at=datetime.now().astimezone().isoformat(timespec="seconds"))
        write_summary(status_path, status)

    update()
    server = None
    try:
        with (work / "server.log").open("a", encoding="utf-8") as server_log:
            server = subprocess.Popen(
                [str(args.server), "-m", str(args.weights), "--host", "127.0.0.1",
                 "--port", str(args.port), "-c", str(args.workers * 4096),
                 "-np", str(args.workers), "--alias", MODEL, "--reasoning", "off",
                 "--cont-batching"],
                stdout=server_log, stderr=subprocess.STDOUT,
            )
            wait_for_model(server, args.port)
            for name, strategy in STAGES:
                if args.pilot_only and name == "full":
                    break
                stage = f"{name}/{strategy}"
                outcome = work / name / f"{strategy}.jsonl"
                cumulative = work / name / f"{strategy}-summary.json"
                update(state="running", active_stage=stage)
                print(f"Running {stage}.", flush=True)
                command = [
                    sys.executable, str(ROOT / "overnight_judge.py"),
                    "--manifest", str(manifests[name]), "--output", str(outcome),
                    "--strategy", strategy, "--port", str(args.port),
                    "--model", MODEL, "--workers", str(args.workers),
                ]
                with (work / name / f"{strategy}.log").open("a", encoding="utf-8") as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
                summary = summarized_stage(manifests[name], outcome, strategy)
                write_summary(cumulative, summary)
                if result.returncode or summary["valid_records"] != summary["expected_records"]:
                    raise RuntimeError(f"Incomplete stage {stage}. See {name}/{strategy}.log and summary.")
                status["completed_stages"].append(stage)
                update(active_stage=None)
                print(
                    f"{stage} saved {summary['valid_records']} validated judgments "
                    f"({summary['reused_records']} reused).",
                    flush=True,
                )
        update(state="complete", milestone_unchanged=digest(milestone) == original_hash)
    except KeyboardInterrupt:
        update(state="interrupted", milestone_unchanged=digest(milestone) == original_hash)
        raise
    except Exception as exc:
        update(state="failed", message=str(exc), milestone_unchanged=digest(milestone) == original_hash)
        raise
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--directory", type=Path, default=ROOT / "overnight")
    parser.add_argument("--port", type=int, default=18131)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.directory.mkdir(parents=True, exist_ok=True)
    with (args.directory / ".run.lock").open("a", encoding="utf-8") as lock:
        try:
            flock(lock, LOCK_EX | LOCK_NB)
        except BlockingIOError:
            parser.error("This batch directory already has a running process.")
        run_batch(args)


if __name__ == "__main__":
    def interrupt(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    main()
