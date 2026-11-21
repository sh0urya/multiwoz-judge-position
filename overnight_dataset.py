"""Build matched, constructed dialogue cases anchored in MultiWOZ test goals."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from zipfile import ZipFile

from experiment import ARCHIVE_HASH, ROOT, digest, read_jsonl, write_jsonl


FAMILIES = ("departure", "destination", "day", "time_cutoff", "party_size")
POSITIONS = ("early", "25%", "middle", "75%", "late")
PILOT_POSITIONS = ("early", "middle", "late")
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
INTRO = (
    "I'm sorting out a few things for a trip. Can we go through them?",
    "Of course. What would you like to sort out?",
)
CLOSING = (
    "That's everything I needed. Thank you!",
    "You're welcome. Safe travels!",
)
# Each independent block has exactly two user-assistant exchanges. Their
# requests provide the information needed to answer without a lookup.
FILLER_BLOCKS = (
    (
        "I'm packing a camera and I would like to keep it handy. Where should it go?",
        "Keep the camera in an accessible pocket of your day bag.",
        "The lens cap comes off easily. How can I avoid losing it?",
        "Put the lens cap in a zipped pocket when you take it off.",
    ),
    (
        "I want to be able to read offline while I travel. What should I do before leaving?",
        "Download the reading to your device before you leave.",
        "How can I check whether the download works?",
        "Open it with your device offline to check that it loads.",
    ),
    (
        "I'm making a paper checklist and want to mark completed tasks. Any suggestion?",
        "Put a small box next to each task so you can tick it off.",
        "I'd also like to mark the important items at a glance.",
        "Draw a star by the items you want to prioritize.",
    ),
    (
        "I plan to carry a reusable bottle and a small umbrella. How should I pack them?",
        "Keep them in accessible pockets so you can take either one out easily.",
        "What can I do to keep the contents of my bag dry after using the umbrella?",
        "Carry a small waterproof sleeve for the wet umbrella.",
    ),
    (
        "I always forget where I put paper receipts. What could help on this trip?",
        "Use one envelope for receipts and keep it in a consistent pocket.",
        "How should I label the envelope?",
        "Write 'Trip receipts' on the front so you can spot it easily.",
    ),
    (
        "I want a simple way to find the photos I take while I'm away.",
        "Make a trip album and add the photos to it as you go.",
        "How should I choose a name for the album?",
        "Use a short name you are likely to remember, such as the name of the trip.",
    ),
    (
        "I'll take a spare pair of glasses. How should I protect them in my bag?",
        "Put the spare glasses in a hard case.",
        "I have a cleaning cloth for them too. Where should that go?",
        "Keep the cloth inside the case so it's with the glasses when you need it.",
    ),
    (
        "I want to keep my charger from tangling with other things in my bag.",
        "Coil the cable and put it in its own small pouch.",
        "How can I recognize the pouch quickly?",
        "Use a distinctive color or add a small label.",
    ),
    (
        "I'm taking a notebook and I'd like to find blank pages quickly.",
        "Use a bookmark to mark the next blank page.",
        "I also want to find notes I've already written.",
        "Keep a simple index on the inside cover.",
    ),
    (
        "I'd like to keep a spare set of clothes separate from everything else.",
        "Put the spare clothes in a packing cube or separate bag.",
        "How do I distinguish clean clothes from ones I've worn?",
        "Keep a separate bag for worn clothes.",
    ),
    (
        "What's a simple way to find my bag among other similar bags?",
        "Add a bright ribbon or distinctive luggage tag.",
        "Can I make the tag recognizable without writing my name on it?",
        "Use a bright shape or symbol you recognize.",
    ),
    (
        "I want to put a few small snacks in my day bag without crushing them.",
        "Use a small rigid container to protect the snacks.",
        "I would like to be able to reach them easily.",
        "Keep the container near the top of your bag.",
    ),
)


@dataclass(frozen=True)
class Source:
    dialogue_id: str
    departure: str
    destination: str
    day: str
    time_field: str
    time_cutoff: str
    party_size: str
    party_size_source: str

    @property
    def values(self):
        return {family: getattr(self, family) for family in FAMILIES}


def load_sources(archive_path):
    if digest(archive_path) != ARCHIVE_HASH:
        raise ValueError("The dataset archive does not match the verified MultiWOZ 2.4 release.")
    with ZipFile(archive_path) as archive:
        dialogues = json.loads(archive.read("MULTIWOZ2.4/data.json"))
        test_ids = archive.read("MULTIWOZ2.4/testListFile.json").decode().split()
        trains = json.loads(archive.read("MULTIWOZ2.4/train_db.json"))
    station_names = sorted({row[field].title() for row in trains for field in ("departure", "destination")})
    development_ids = {row["dialogue_id"] for row in read_jsonl(ROOT / "data" / "pilot.jsonl")}
    sources = []
    seen_goals = set()
    for dialogue_id in sorted(test_ids):
        dialogue = dialogues[dialogue_id]
        if dialogue_id in development_ids or len(dialogue["log"]) < 20:
            continue
        goal = dialogue["goal"].get("train") or {}
        info = goal.get("info") or {}
        if not all(info.get(field) for field in ("departure", "destination", "day")):
            continue
        time_field = next((field for field in ("arriveBy", "leaveAt") if info.get(field)), None)
        if time_field is None:
            continue
        cutoff = info[time_field]
        day = info["day"].title()
        departure = info["departure"].title()
        destination = info["destination"].title()
        if day not in DAYS or departure == destination or not all(
            station in station_names for station in (departure, destination)
        ):
            continue
        try:
            hour, minute = map(int, cutoff.split(":"))
        except (AttributeError, ValueError):
            continue
        if not (0 <= hour < 24 and 0 <= minute < 60):
            continue
        cutoff = f"{hour:02d}:{minute:02d}"
        count = str((goal.get("book") or {}).get("people", ""))
        original_count = count.isdecimal() and int(count) > 0
        if not original_count:
            count = str(random.Random(dialogue_id).randint(2, 5))
        source = Source(
            dialogue_id, departure, destination, day, time_field, cutoff, count,
            "train.book.people" if original_count else "constructed from source ID",
        )
        goal_key = (source.time_field, *source.values.values())
        if goal_key in seen_goals:
            continue
        seen_goals.add(goal_key)
        sources.append(source)
    random.Random(42).shuffle(sources)
    if len(sources) < 46:
        raise ValueError(f"Found only {len(sources)} eligible distinct source goals. Expected at least 46.")
    return sources, station_names


def alternatives(source, stations, scenario_index):
    rng = random.Random(source.dialogue_id)
    eligible_stations = [name for name in stations if name not in (source.departure, source.destination)]
    hour, minute = map(int, source.time_cutoff.split(":"))
    delta = 30 if scenario_index % 2 == 0 else -30
    alternative_minutes = hour * 60 + minute + delta
    if not 0 <= alternative_minutes < 24 * 60:
        alternative_minutes -= 2 * delta
    alt_hour, alt_minute = divmod(alternative_minutes, 60)
    count = int(source.party_size)
    return {
        "departure": rng.choice(eligible_stations),
        "destination": rng.choice(eligible_stations),
        "day": DAYS[(DAYS.index(source.day) + (1 if scenario_index % 2 == 0 else -1)) % len(DAYS)],
        "time_cutoff": f"{alt_hour:02d}:{alt_minute:02d}",
        "party_size": str(count + 1 if count == 1 or scenario_index % 2 == 0 else count - 1),
    }


def rail_block(source, reported_values):
    user_time = "arrive by" if source.time_field == "arriveBy" else "leave at or after"
    readback_time = "arriving by" if source.time_field == "arriveBy" else "leaving at or after"
    stated = (
        f"For the train search, I'm leaving from {source.departure} and going to "
        f"{source.destination} on {source.day}. I need to {user_time} {source.time_cutoff}, "
        f"and the number of travelers is {source.party_size}."
    )
    reply = (
        f"I'll look for a train from {reported_values['departure']} to {reported_values['destination']} "
        f"on {reported_values['day']}, {readback_time} {reported_values['time_cutoff']}. "
        f"The number of travelers is {reported_values['party_size']}."
    )
    return (
        stated,
        "I've noted your train requirements and can read them back before searching.",
        "Could you read back the route, day, time requirement, and group size for me?",
        reply,
    )


def assemble(source, values, fillers, slot):
    blocks = list(fillers)
    blocks.insert(slot, rail_block(source, values))
    return list(INTRO) + [message for block in blocks for message in block] + list(CLOSING)


def build_cases(cohort, sources, station_names):
    is_pilot = cohort == "pilot"
    positions = PILOT_POSITIONS if is_pilot else POSITIONS
    cases = []
    for scenario_index, source in enumerate(sources):
        families = (("destination",) if scenario_index < 3 else ("day",)) if is_pilot else FAMILIES
        wrong = alternatives(source, station_names, scenario_index)
        # Hold the distractor order fixed across every position for this source.
        # Select different sets and orders across sources to vary the contexts.
        fillers = random.Random(source.dialogue_id).sample(FILLER_BLOCKS, 4)
        for position in positions:
            slot = POSITIONS.index(position)
            expected_turn = 3 + 2 * slot
            correct = assemble(source, source.values, fillers, slot)
            for family in families:
                pair_id = f"{cohort}:{source.dialogue_id}:{family}:{position}"
                incorrect = assemble(source, {**source.values, family: wrong[family]}, fillers, slot)
                common = {
                    "dialogue_id": source.dialogue_id,
                    "family": family,
                    "position": position,
                    "pair_id": pair_id,
                    "evidence_turn": expected_turn - 1,
                    "expected_turn": expected_turn,
                    "correct_value": source.values[family],
                    "incorrect_value": wrong[family],
                }
                for condition, messages in (("correct", correct), ("incorrect", incorrect)):
                    cases.append({
                        **common,
                        "case_id": f"{pair_id}:{condition}",
                        "condition": condition,
                        "messages": messages,
                    })
    validate_cases(cohort, cases, sources, station_names)
    return cases


def validate_cases(cohort, cases, sources, station_names):
    expected_pairs = 18 if cohort == "pilot" else 1000
    ids = {case["case_id"] for case in cases}
    if len(cases) != 2 * expected_pairs or len(ids) != len(cases):
        raise ValueError(f"{cohort} has {len(cases)} cases and {len(ids)} unique IDs.")
    pairs = defaultdict(dict)
    for case in cases:
        pairs[case["pair_id"]][case["condition"]] = case
    if len(pairs) != expected_pairs:
        raise ValueError(f"{cohort} has {len(pairs)} distinct pairs.")
    source_info = {
        source.dialogue_id: (source, alternatives(source, station_names, index))
        for index, source in enumerate(sources)
    }
    for pair in pairs.values():
        correct, incorrect = pair["correct"], pair["incorrect"]
        original, edited = correct["messages"], incorrect["messages"]
        target = 2 * correct["expected_turn"] - 1
        evidence = 2 * (correct["evidence_turn"] - 1)
        source, wrong = source_info[correct["dialogue_id"]]
        correct_block = rail_block(source, source.values)
        edited_block = rail_block(source, {**source.values, correct["family"]: wrong[correct["family"]]})
        if (
            len(original) != 24 or len(edited) != len(original)
            or original[:target] != edited[:target] or original[target + 1:] != edited[target + 1:]
            or original[target] == edited[target] or evidence != target - 3
            or original[evidence:target + 1] != list(correct_block)
            or edited[evidence:target + 1] != list(edited_block)
            or correct["correct_value"] == correct["incorrect_value"]
        ):
            raise ValueError(f"Malformed matched pair: {correct['pair_id']}")


def write_pilot_review(path, review_packet):
    lines = [
        "# Blind pilot review",
        "For each transcript, record any reply that contradicts the user's requirements. "
        "Cite the reply and the supporting user turn. Also note any awkward exchange "
        "or later correction that would affect judging.",
    ]
    for entry in review_packet:
        lines.append(f"## {entry['review_id']}")
        for index, message in enumerate(entry["messages"]):
            speaker = "User" if index % 2 == 0 else "Assistant"
            lines.append(f"**{speaker} {index // 2 + 1}** {message}")
    path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def save_cohort(out_dir, cohort, sources, cases):
    directory = out_dir / cohort
    write_jsonl(directory / "cases.jsonl", cases)
    order = cases.copy()
    random.Random(f"review:{cohort}:42").shuffle(order)
    review_packet = []
    review_key = []
    for case in order:
        review_id = hashlib.sha256(case["case_id"].encode()).hexdigest()[:12]
        review_packet.append({"review_id": review_id, "messages": case["messages"]})
        review_key.append({"review_id": review_id, **{k: v for k, v in case.items() if k != "messages"}})
    if len(review_packet) != len({entry["review_id"] for entry in review_packet}):
        raise ValueError("Duplicate blind review identifier.")
    write_jsonl(directory / "review" / "transcripts.jsonl", review_packet)
    if cohort == "pilot":
        write_pilot_review(directory / "review" / "transcripts.md", review_packet)
    write_jsonl(directory / "review_key.jsonl", review_key)
    provenance = []
    for source in sources:
        provenance.append({
            "dialogue_id": source.dialogue_id,
            "source_split": "test",
            "source_fields": {
                "departure": "train.info.departure",
                "destination": "train.info.destination",
                "day": "train.info.day",
                "time_cutoff": f"train.info.{source.time_field}",
                "party_size": source.party_size_source,
            },
            "correct_values": source.values,
            "transcript_construction": "Independent constructed travel advice blocks and train readback. "
            "User train requirements are adapted from the source goal; wording is constructed throughout.",
        })
    write_jsonl(directory / "provenance.jsonl", provenance)
    groups = Counter(case["condition"] for case in cases)
    unique_texts = len({tuple(case["messages"]) for case in cases})
    print(f"{cohort}: {len(sources)} base scenarios, {groups['incorrect']} edited cases, "
          f"{groups['correct']} matched controls, {unique_texts} unique transcripts.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("pilot", "full", "all"), default="all")
    parser.add_argument("--archive", type=Path, default=ROOT / "data" / "MULTIWOZ2.4.zip")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "overnight")
    args = parser.parse_args()
    sources, station_names = load_sources(args.archive)
    selections = {"pilot": sources[:6], "full": sources[6:46]}
    for cohort, selected in selections.items():
        if args.cohort in (cohort, "all"):
            cases = build_cases(cohort, selected, station_names)
            save_cohort(args.out_dir, cohort, selected, cases)


if __name__ == "__main__":
    main()
