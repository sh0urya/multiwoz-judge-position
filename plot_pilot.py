"""Plot the matched Qwen3.5-9B milestone results."""

import json
from statistics import mean

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from experiment import POSITIONS, ROOT, dialogue_judgments, read_jsonl


cases = read_jsonl(ROOT / "data" / "pilot.jsonl")
clean_ids = [case["dialogue_id"] for case in cases if case["position"] == "clean"]
figure, (detection_axis, score_axis) = plt.subplots(
    2, 1, sharex=True, figsize=(3.35, 3.15), gridspec_kw={"hspace": 0.32}
)
for strategy, label, color, marker, linestyle in (
    ("j1", "J1 · Whole dialogue", "#225b91", "o", "-"),
    ("j3", "J3 · Lowest turn", "#a85318", "x", "--"),
):
    judgments = dialogue_judgments(
        cases, read_jsonl(ROOT / "results" / f"{strategy}-qwen-9b.jsonl")
    )
    reviews = json.loads(
        (ROOT / "results" / f"review-{strategy}-qwen-9b.json").read_text(encoding="utf-8")
    )
    detections = [sum(reviews[c][position] for c in clean_ids) for position in POSITIONS]
    score_drops = [
        mean(judgments[c, "clean"]["score"] - judgments[c, position]["score"] for c in clean_ids)
        for position in POSITIONS
    ]
    style = {
        "color": color,
        "marker": marker,
        "markerfacecolor": "white",
        "linestyle": linestyle,
        "label": label,
    }
    detection_axis.plot(POSITIONS, detections, **style)
    score_axis.plot(POSITIONS, score_drops, **style)

detection_axis.set_ylabel("Errors detected")
detection_axis.set_yticks(range(len(clean_ids) + 1))
detection_axis.set_ylim(-0.2, len(clean_ids) + 0.3)
score_axis.set_ylabel("Mean drop (points)")
score_axis.set_xlabel("Error position")
score_axis.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
score_axis.axhline(0, color="#666666", linewidth=0.7, zorder=0)
for axis in (detection_axis, score_axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8)
    axis.yaxis.label.set_size(8)
    axis.xaxis.label.set_size(8)
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.5)
    axis.set_axisbelow(True)
handles, labels = detection_axis.get_legend_handles_labels()
figure.legend(handles, labels, loc="upper center", ncols=2, frameon=False, fontsize=8)
figure.subplots_adjust(left=0.19, right=0.97, top=0.88, bottom=0.13)
figure.savefig(ROOT / "milestone" / "position-curve.pdf")
