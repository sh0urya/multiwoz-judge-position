"""Build the correct-route counterparts to the original route-error pilot."""

from experiment import ROOT, read_jsonl, write_jsonl


CORRECT_LOCATIONS = {
    "PMUL0509.json": "Norwich",
    "MUL1569.json": "London Kings Cross",
    "PMUL1801.json": "Peterborough",
}


def main():
    cases = read_jsonl(ROOT / "data" / "pilot.jsonl")
    originals = {
        case["dialogue_id"]: case["messages"]
        for case in cases
        if case["position"] == "clean"
    }
    controls = []
    for case in cases:
        if case["position"] == "clean":
            continue
        dialogue_id = case["dialogue_id"]
        original = originals[dialogue_id]
        target = 2 * case["expected_turn"] - 1
        expected = original.copy()
        false_claim = case["false_claim"]
        expected[target] += f" {false_claim}"
        if expected != case["messages"] or false_claim.count("Ely") != 1:
            raise ValueError(f"Unexpected pilot edit in {dialogue_id}, {case['position']}")

        correct_claim = false_claim.replace("Ely", CORRECT_LOCATIONS[dialogue_id])
        messages = original.copy()
        messages[target] += f" {correct_claim}"
        controls.append(
            {
                "dialogue_id": dialogue_id,
                "position": case["position"],
                "condition": "correct_route_append",
                "expected_turn": case["expected_turn"],
                "correct_claim": correct_claim,
                "messages": messages,
            }
        )

    path = ROOT / "data" / "route_controls.jsonl"
    write_jsonl(path, controls)
    print(f"Wrote {len(controls)} correct-route counterparts to {path.name}.")


if __name__ == "__main__":
    main()
