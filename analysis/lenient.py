import json, sys, random
from collections import defaultdict
from pathlib import Path
FULL=Path(sys.argv[1]); POS=["early","25%","middle","75%","late"]
jl=lambda p:[json.loads(l) for l in open(p,encoding='utf-8') if l.strip()]
nrm=lambda v:" ".join(str(v or "").casefold().strip().strip(" .!?\"'").split())
cases={c["case_id"]:c for c in jl(FULL/"cases.jsonl")}
j1={r["case_id"]:r["judgment"] for r in jl(FULL/"j1.jsonl")}
j3=defaultdict(dict)
for r in jl(FULL/"j3.jsonl"): j3[r["case_id"]][r["focus_turn"]]=r["judgment"]
def issues(cid,judge):
    if judge=="J1": return j1[cid]["issues"]
    return [i for t,jd in j3[cid].items() for i in jd["issues"]]
def contains(a,b): return nrm(b) in nrm(a)
out={}
for judge in ("J1","J3"):
    rows=[]
    carry=total_after=0
    for cid,c in cases.items():
        if c["condition"]!="incorrect": continue
        tgt=[i for i in issues(cid,judge) if i["turn"]==c["expected_turn"]]
        len_hit=any(i.get("supporting_user_turn")==c["evidence_turn"] and contains(i.get("observed_value"),c["incorrect_value"]) and contains(i.get("expected_value"),c["correct_value"]) for i in tgt)
        val_hit=any(contains(i.get("observed_value"),c["incorrect_value"]) and contains(i.get("expected_value"),c["correct_value"]) for i in tgt)
        rows.append((c["dialogue_id"],c["position"],c["family"],len_hit,val_hit))
        for i in issues(cid,judge):
            if i["turn"]>c["expected_turn"]:
                total_after+=1
                carry+=contains(i.get("observed_value"),c["incorrect_value"])
    by=lambda k,pos:sum(r[k] for r in rows if r[1]==pos)
    out[judge]={"values+evidence (contains)":{p:by(3,p) for p in POS},"values only (contains)":{p:by(4,p) for p in POS},
        "totals":[sum(r[3] for r in rows),sum(r[4] for r in rows)],
        "later_flags":total_after,"later_flags_quoting_the_wrong_value":carry,
        "by_family_values+evidence":{f:sum(r[3] for r in rows if r[2]==f) for f in ["departure","destination","day","time_cutoff","party_size"]}}
    # bootstrap late-early for values+evidence
    ds=sorted({r[0] for r in rows}); byd=defaultdict(list)
    for r in rows: byd[r[0]].append(r)
    rng=random.Random(3); vals=[]
    for _ in range(4000):
        S=[x for _ in ds for x in byd[rng.choice(ds)]]
        f=lambda p:sum(x[3] for x in S if x[1]==p)/max(1,sum(1 for x in S if x[1]==p))
        vals.append(f("late")-f("early"))
    vals.sort(); out[judge]["late_minus_early_values+evidence"]=[round(by(3,"late")/200-by(3,"early")/200,3),round(vals[100],3),round(vals[3899],3)]
print(json.dumps(out,indent=1))
