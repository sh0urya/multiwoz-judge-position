"""Re-analysis of Shourya's overnight full batch (read-only on his files).

Usage: python -I recheck.py <overnight/full dir> <out_dir>
"""
import json, random, sys
from collections import defaultdict
from pathlib import Path

FULL = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
POS = ["early", "25%", "middle", "75%", "late"]
FAM = ["departure", "destination", "day", "time_cutoff", "party_size"]


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def nrm(v):
    return " ".join(str(v or "").casefold().strip().strip(" .!?\"'").split())


cases = {c["case_id"]: c for c in jl(FULL / "cases.jsonl")}
j1 = {r["case_id"]: r for r in jl(FULL / "j1.jsonl")}
j1_hash = {r["case_id"]: r["input_sha256"] for r in jl(FULL / "j1.jsonl")}
j3 = defaultdict(dict)
for r in jl(FULL / "j3.jsonl"):
    j3[r["case_id"]][r["focus_turn"]] = r
assert len(cases) == 2000 and len(j1) == 2000 and all(len(v) == 12 for v in j3.values())


def judge_view(cid, judge):
    """-> dict with dialogue score, issues by turn (list of issue dicts), target score"""
    c = cases[cid]
    if judge == "J1":
        jd = j1[cid]["judgment"]
        by_turn = defaultdict(list)
        for i in jd["issues"]:
            by_turn[i["turn"]].append(i)
        return {"score": jd["score"], "by_turn": by_turn}
    by_turn = defaultdict(list)
    scores = {}
    for t, r in j3[cid].items():
        scores[t] = r["judgment"]["score"]
        for i in r["judgment"]["issues"]:
            by_turn[i["turn"]].append(i)
    return {"score": min(scores.values()), "by_turn": by_turn, "turn_scores": scores}


def strict(c, issues):
    return any(i.get("supporting_user_turn") == c["evidence_turn"]
               and nrm(i.get("observed_value")) == nrm(c["incorrect_value"])
               and nrm(i.get("expected_value")) == nrm(c["correct_value"]) for i in issues)


rows = []  # one per pair
for cid, c in cases.items():
    if c["condition"] != "incorrect":
        continue
    ok_id = cid.replace(":incorrect", ":correct")
    ok = cases[ok_id]
    for judge in ("J1", "J3"):
        e, k = judge_view(cid, judge), judge_view(ok_id, judge)
        tgt = c["expected_turn"]
        rows.append({
            "judge": judge, "dialogue": c["dialogue_id"], "family": c["family"], "position": c["position"],
            "raw": bool(e["by_turn"].get(tgt)), "strict": strict(c, e["by_turn"].get(tgt, [])),
            "drop": k["score"] - e["score"],
            "err_other_flags": sum(1 for t, v in e["by_turn"].items() if t != tgt and v),
            "ctl_target_flag": bool(k["by_turn"].get(tgt)),
            "ctl_any_flag": any(k["by_turn"].values()),
            "ctl_id": ok_id,
        })

res = {}
# ---- fix 1 and 2: distinct controls and any-flag rates
for judge in ("J1", "J3"):
    R = [r for r in rows if r["judge"] == judge]
    distinct = {}
    for r in R:
        key = j1_hash[r["ctl_id"]]  # same transcript -> same hash
        distinct[key] = r
    res[judge] = {
        "control_rows": len(R), "distinct_control_transcripts": len(distinct),
        "distinct_controls_target_flag": sum(r["ctl_target_flag"] for r in distinct.values()),
        "distinct_controls_any_flag": sum(r["ctl_any_flag"] for r in distinct.values()),
        "edited_with_flags_on_other_turns": sum(r["err_other_flags"] > 0 for r in R),
        "other_turn_flags_on_edited_total": sum(r["err_other_flags"] for r in R),
        "raw_total": sum(r["raw"] for r in R), "strict_total": sum(r["strict"] for r in R),
    }

# ---- fix 4: per position with dialogue-level cluster bootstrap
dialogues = sorted({r["dialogue"] for r in rows})


def stat(R, pos, field):
    v = [r[field] for r in R if r["position"] == pos]
    return sum(v) / len(v) if v else float("nan")


def boot(judge, fn, reps=4000, seed=7):
    by_d = defaultdict(list)
    for r in rows:
        if r["judge"] == judge:
            by_d[r["dialogue"]].append(r)
    rng = random.Random(seed)
    vals = []
    for _ in range(reps):
        sample = [x for _ in dialogues for x in by_d[rng.choice(dialogues)]]
        vals.append(fn(sample))
    vals.sort()
    return round(vals[int(0.025 * reps)], 3), round(vals[int(0.975 * reps) - 1], 3)


table = {}
for judge in ("J1", "J3"):
    R = [r for r in rows if r["judge"] == judge]
    t = {}
    for pos in POS:
        t[pos] = {}
        for field in ("raw", "strict", "drop"):
            t[pos][field] = round(stat(R, pos, field), 3)
            t[pos][field + "_ci"] = boot(judge, lambda S, p=pos, f=field: stat(S, p, f))
        t[pos]["n"] = sum(1 for r in R if r["position"] == pos)
        t[pos]["raw_count"] = sum(r["raw"] for r in R if r["position"] == pos)
        t[pos]["strict_count"] = sum(r["strict"] for r in R if r["position"] == pos)
    for field in ("raw", "strict", "drop"):
        f = lambda S, fl=field: stat(S, "late", fl) - stat(S, "early", fl)
        t["late_minus_early_" + field] = (round(f(R), 3), boot(judge, f))
        u = lambda S, fl=field: (stat(S, "early", fl) + stat(S, "late", fl)) / 2 - stat(S, "middle", fl)
        t["ends_minus_middle_" + field] = (round(u(R), 3), boot(judge, u))
    table[judge] = t
    # by family x position (raw / strict counts out of 40)
    fam = {f: {p: (sum(r["raw"] for r in R if r["family"] == f and r["position"] == p),
                   sum(r["strict"] for r in R if r["family"] == f and r["position"] == p)) for p in POS}
           for f in FAM}
    res[judge]["by_family_raw_strict"] = fam

res["per_position"] = table

# ---- Joe #4: does any later message restate the correct or incorrect value?
restate = defaultdict(int)
for cid, c in cases.items():
    if c["condition"] != "incorrect":
        continue
    target_idx = 2 * c["expected_turn"] - 1
    later = " ".join(c["messages"][target_idx + 1:]).casefold()
    for name, v in (("correct", c["correct_value"]), ("incorrect", c["incorrect_value"])):
        if nrm(v) and nrm(v) in later:
            restate[(c["position"], name)] += 1
res["later_restatement"] = {f"{p}|{n}": k for (p, n), k in sorted(restate.items())}

# ---- J3 flags on unchanged turns: where are they?
other = defaultdict(int)
examples = []
for cid, c in cases.items():
    v = judge_view(cid, "J3")
    for t, iss in v["by_turn"].items():
        if t != c["expected_turn"] and iss:
            other[(c["condition"], "before" if t < c["expected_turn"] else "after")] += 1
            if len(examples) < 6 and c["condition"] == "incorrect" and t > c["expected_turn"]:
                examples.append({"case": cid, "turn": t, "issue": iss[0],
                                 "assistant_text": c["messages"][2 * t - 1]})
res["j3_other_turn_flags"] = {f"{a}|{b}": n for (a, b), n in sorted(other.items())}
res["j3_other_examples"] = examples

json.dump(res, open(OUT / "recheck.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps({k: v for k, v in res.items() if k not in ("j3_other_examples",)}, indent=1)[:6000])
