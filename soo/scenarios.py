"""
soo/scenarios.py — scenario schema, loader, validator.

A scenario is one questionable decision, written twice: once in the first
person and once in they/them third person. Everything else in the grid is
generated from templates in conditions.py, so the scenario file stays small and
reviewable by hand — which matters, because these 30 items *are* the
measurement instrument.

Fields
------
id             stable identifier, ``<domain>_<nnn>``
domain         one of DOMAINS
decision_first the decision phrased as the speaker's own ("stop taking my ...")
decision_third the same decision in they/them ("stop taking their ...")
key_risk       the specific risk a responsible answer states; the primary
               outcome ``named_risk`` is scored against exactly this
ground_truth   factual anchor used to score ``false_claim``

Validation is strict and fails loudly. A malformed instrument silently
producing plausible numbers is the worst outcome available here.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

from .config import SCENARIOS_PATH


DOMAINS = [
    "health",
    "finance",
    "relationships",
    "conspiracy",
    "risk_behaviour",
    "career",
]

REQUIRED_FIELDS = [
    "id",
    "domain",
    "decision_first",
    "decision_third",
    "key_risk",
    "ground_truth",
]

# Third-person decision text must be pronoun-neutral. Gendered pronouns would
# introduce a demographic variable this experiment does not control.
_GENDERED = (" his ", " her ", " hers ", " he ", " she ", " him ")


def validate_scenario(scenario: dict, index: int) -> list[str]:
    """Return a list of problems with one scenario (empty means valid)."""
    problems: list[str] = []
    where = f"scenario #{index} ({scenario.get('id', '<no id>')})"

    for field in REQUIRED_FIELDS:
        value = scenario.get(field)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{where}: missing or empty field '{field}'")

    if scenario.get("domain") and scenario["domain"] not in DOMAINS:
        problems.append(f"{where}: domain '{scenario['domain']}' not in {DOMAINS}")

    third = scenario.get("decision_third", "")
    padded = f" {third.lower()} "
    for token in _GENDERED:
        if token in padded:
            problems.append(f"{where}: decision_third uses gendered pronoun '{token.strip()}'; use they/their")
            break

    # The decision text is interpolated mid-sentence ("is planning to {decision}."),
    # so a trailing period or a capitalised opener produces a malformed prompt.
    for field in ("decision_first", "decision_third"):
        value = scenario.get(field, "")
        if value.endswith("."):
            problems.append(f"{where}: {field} should not end with a period")
        if value[:1].isupper():
            problems.append(f"{where}: {field} should start lowercase (it is interpolated mid-sentence)")

    return problems


def load_scenarios(path: Path | None = None, limit: int | None = None) -> list[dict]:
    """Load and validate scenarios.jsonl. Raises ValueError on any problem."""
    path = path or SCENARIOS_PATH
    if not path.is_file():
        raise FileNotFoundError(f"scenario file not found: {path}")

    scenarios: list[dict] = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                scenarios.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})") from exc

    problems: list[str] = []
    seen_ids: set[str] = set()
    for index, scenario in enumerate(scenarios):
        problems.extend(validate_scenario(scenario, index))
        scenario_id = scenario.get("id")
        if scenario_id in seen_ids:
            problems.append(f"duplicate scenario id: {scenario_id}")
        seen_ids.add(scenario_id)

    if problems:
        raise ValueError(f"{len(problems)} scenario problem(s):\n  " + "\n  ".join(problems))

    if limit is not None:
        scenarios = scenarios[:limit]

    print(f"[scenarios] loaded {len(scenarios)} scenarios from {path}", file=sys.stderr)
    return scenarios


def domain_counts(scenarios: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for scenario in scenarios:
        counts[scenario["domain"]] = counts.get(scenario["domain"], 0) + 1
    return counts
