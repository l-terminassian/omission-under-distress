"""
soo/validate.py — is the judge measuring what we say it measures?

Every headline number in this study is produced by an LLM judge scoring subtle
constructs ("did it discourage outside help?"). If the judge does not track what
a careful human would call the same thing, the numbers are decoration, so this
check runs before any claim is made.

Flow
----
1. ``soo validate-export`` writes a CSV of sampled responses with blank rubric
   columns, in shuffled order and with no condition labels or judge scores
   visible — so hand-labelling cannot be anchored by either.
2. You fill in the blanks.
3. ``soo validate-score`` reports Cohen's kappa per rubric item.

**Pre-registered decision rule:** any item below ``KAPPA_FLOOR`` (0.60) is
dropped from the headline claims and reported as unreliable. The rule is fixed
in config.py before the numbers exist, so it cannot be negotiated afterwards.
"""
from __future__ import annotations

import csv
import random
import sys

from .config import (
    ADVICE_SOURCES,
    KAPPA_FLOOR,
    MODELS_UNDER_TEST,
    PRIMARY_TURN,
    RESPONSES_PATH,
    SCORES_PATH,
    VALIDATION_LABELLED_PATH,
    VALIDATION_SAMPLE_PATH,
    VALIDATION_SAMPLE_SIZE,
    VALIDATION_SEED,
)
from .rubric import RUBRIC_ITEMS, items_for_turn
from .scenarios import load_scenarios
from .store import read_jsonl


def cohen_kappa(a: list[int], b: list[int]) -> float | None:
    """Cohen's kappa for two aligned label sequences.

    Returns None when undefined. The degenerate case is worth understanding:
    when both raters use exactly one category for everything, expected agreement
    is 1.0 and kappa is 0/0. That is not "perfect agreement" — it means the item
    has no variance in this sample and carries no information, so it must not be
    reported as if it passed.
    """
    if len(a) != len(b) or not a:
        return None
    categories = sorted(set(a) | set(b))
    if len(categories) < 2:
        return None

    n = len(a)
    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    expected = sum((a.count(c) / n) * (b.count(c) / n) for c in categories)
    if abs(1.0 - expected) < 1e-12:
        return None
    return (observed - expected) / (1.0 - expected)


def export_for_labelling(n: int | None = None) -> None:
    """Write a stratified, shuffled sample for hand-labelling."""
    n = n or VALIDATION_SAMPLE_SIZE
    responses = {r["record_id"]: r for r in read_jsonl(RESPONSES_PATH)}
    scores = [s for s in read_jsonl(SCORES_PATH) if s.get("scores")]
    if not scores:
        raise RuntimeError(f"no scored rows at {SCORES_PATH}; run `soo judge` first")
    scenarios = {s["id"]: s for s in load_scenarios()}

    # Stratify so the sample spans the design rather than clumping in one corner.
    strata: dict[tuple, list[dict]] = {}
    for row in scores:
        key = (row["model_key"], row["attribution"], row["vulnerability"],
               row.get("advice_source"), row["turn"])
        strata.setdefault(key, []).append(row)

    rng = random.Random(VALIDATION_SEED)
    per_stratum = max(1, n // max(len(strata), 1))
    sampled: list[dict] = []
    for key in sorted(strata):
        bucket = strata[key]
        sampled.extend(rng.sample(bucket, min(per_stratum, len(bucket))))
    rng.shuffle(sampled)
    sampled = sampled[:n]

    VALIDATION_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_items = list(RUBRIC_ITEMS)
    header = ["row_id", "score_record_id", "turn", "scenario_id", "key_risk", "ground_truth", "user_message",
              "assistant_response"] + all_items

    with VALIDATION_SAMPLE_PATH.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index, row in enumerate(sampled):
            response = responses.get(row["response_record_id"], {})
            scenario = scenarios.get(row["scenario_id"], {})
            turn = row["turn"]
            user_message = response.get("prompt", "") if turn == 1 else response.get("pushback", "")
            assistant = response.get(f"response_turn{turn}", "")
            applicable = set(items_for_turn(turn))
            # "n/a" for items that do not apply to this turn, blank for you to fill.
            cells = ["" if item in applicable else "n/a" for item in all_items]
            writer.writerow(
                [index, row["record_id"], turn, row["scenario_id"], scenario.get("key_risk", ""),
                 scenario.get("ground_truth", ""), user_message, assistant] + cells
            )

    print(f"[validate] wrote {len(sampled)} rows to {VALIDATION_SAMPLE_PATH}", file=sys.stderr)
    print(
        "[validate] Fill in the blank rubric columns using the same definitions the judge saw "
        "(see rubric.py), then run `soo validate-score`.\n"
        "[validate] Condition labels and judge scores are deliberately absent from this file so "
        "your labelling cannot be anchored by either.",
        file=sys.stderr,
    )


def extend_for_labelling(n: int = 150) -> None:
    """Append a second labelling batch below the rows already in the file.

    Two differences from ``export_for_labelling``, both deliberate:

    * **Restricted to the neutral/distressed contrast at turn 1.** The first
      batch was stratified across the whole design, which is the right shape for
      validating an *instrument*. This batch exists to test a specific *claim* —
      that risk-naming falls and referral rises between those two cues — and the
      two middle levels contribute almost nothing to that contrast. Sampling only
      the extremes buys roughly the power of twice as many uniform labels.
    * **Appends rather than overwrites.** Rows already present keep their
      ``row_id`` and any labels already entered; new rows continue the numbering.
      Re-running is safe: already-sampled responses are never drawn twice.

    Blindness is preserved exactly as in the first batch — no condition column,
    no judge score, shuffled order.
    """
    existing_rows: list[dict] = []
    if VALIDATION_SAMPLE_PATH.is_file():
        with VALIDATION_SAMPLE_PATH.open() as handle:
            existing_rows = list(csv.DictReader(handle))
    seen = {r["score_record_id"] for r in existing_rows}
    next_id = max((int(r["row_id"]) for r in existing_rows), default=-1) + 1

    responses = {r["record_id"]: r for r in read_jsonl(RESPONSES_PATH)}
    scenarios = {s["id"]: s for s in load_scenarios()}
    factorial = {m["key"] for m in MODELS_UNDER_TEST}

    pool = [
        s
        for s in read_jsonl(SCORES_PATH)
        if s.get("scores")
        and s["record_id"] not in seen
        and s["turn"] == PRIMARY_TURN
        and s["model_key"] in factorial
        and s.get("advice_source") in ADVICE_SOURCES
        and s["vulnerability"] in ("neutral", "distressed")
    ]
    if not pool:
        raise RuntimeError("no unlabelled rows left in the neutral/distressed contrast")

    # Balance the two cues, then across models, so the contrast is not carried by
    # one arm or one vendor. Unbalanced labels would make the human estimate of
    # the gradient a function of which cells happened to be drawn.
    strata: dict[tuple, list[dict]] = {}
    for row in pool:
        strata.setdefault((row["vulnerability"], row["model_key"]), []).append(row)

    rng = random.Random(VALIDATION_SEED + 1)
    per_stratum = max(1, n // max(len(strata), 1))
    sampled: list[dict] = []
    for key in sorted(strata):
        bucket = strata[key]
        sampled.extend(rng.sample(bucket, min(per_stratum, len(bucket))))
    # Top up from whatever is left if integer division under-filled the quota.
    if len(sampled) < n:
        chosen = {r["record_id"] for r in sampled}
        remainder = [r for r in pool if r["record_id"] not in chosen]
        rng.shuffle(remainder)
        sampled.extend(remainder[: n - len(sampled)])
    rng.shuffle(sampled)
    sampled = sampled[:n]

    all_items = list(RUBRIC_ITEMS)
    header = ["row_id", "score_record_id", "turn", "scenario_id", "key_risk", "ground_truth", "user_message",
              "assistant_response"] + all_items

    with VALIDATION_SAMPLE_PATH.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in existing_rows:
            writer.writerow([row.get(column, "") for column in header])
        for offset, row in enumerate(sampled):
            response = responses.get(row["response_record_id"], {})
            scenario = scenarios.get(row["scenario_id"], {})
            turn = row["turn"]
            user_message = response.get("prompt", "") if turn == 1 else response.get("pushback", "")
            assistant = response.get(f"response_turn{turn}", "")
            applicable = set(items_for_turn(turn))
            cells = ["" if item in applicable else "n/a" for item in all_items]
            writer.writerow(
                [next_id + offset, row["record_id"], turn, row["scenario_id"], scenario.get("key_risk", ""),
                 scenario.get("ground_truth", ""), user_message, assistant] + cells
            )

    print(
        f"[validate] kept {len(existing_rows)} existing rows, appended {len(sampled)} new ones "
        f"(row_id {next_id}-{next_id + len(sampled) - 1}) in {VALIDATION_SAMPLE_PATH}",
        file=sys.stderr,
    )
    print(
        "[validate] New rows are turn-1, neutral-vs-distressed only. Fill the blank rubric columns, "
        f"then copy to {VALIDATION_LABELLED_PATH.name} and run `soo validate-score`.",
        file=sys.stderr,
    )


def score_agreement() -> dict:
    """Judge-vs-human Cohen's kappa per rubric item, with the pass/fail rule applied."""
    if not VALIDATION_LABELLED_PATH.is_file():
        raise RuntimeError(
            f"no labelled file at {VALIDATION_LABELLED_PATH}. "
            f"Fill in {VALIDATION_SAMPLE_PATH.name}, save it as {VALIDATION_LABELLED_PATH.name}, and re-run."
        )

    judge_by_id = {s["record_id"]: s for s in read_jsonl(SCORES_PATH) if s.get("scores")}

    human_rows: list[dict] = []
    with VALIDATION_LABELLED_PATH.open() as handle:
        human_rows = list(csv.DictReader(handle))

    results: dict[str, dict] = {}
    for item in RUBRIC_ITEMS:
        human_labels: list[int] = []
        judge_labels: list[int] = []
        for row in human_rows:
            raw = (row.get(item) or "").strip()
            if raw in ("", "n/a"):
                continue
            judged = judge_by_id.get(row["score_record_id"])
            if not judged or item not in (judged.get("scores") or {}):
                continue
            try:
                human_labels.append(int(raw))
            except ValueError:
                print(f"[validate] row {row.get('row_id')}: unparseable '{item}' value {raw!r}", file=sys.stderr)
                continue
            judge_labels.append(int(judged["scores"][item]))

        kappa = cohen_kappa(human_labels, judge_labels)
        agreement = (
            sum(1 for h, j in zip(human_labels, judge_labels, strict=True) if h == j) / len(human_labels)
            if human_labels
            else None
        )
        results[item] = {"kappa": kappa, "raw_agreement": agreement, "n": len(human_labels)}

    print("\n[validate] judge-vs-human agreement", file=sys.stderr)
    print(f"[validate] {'item':<28} {'n':>5} {'raw':>7} {'kappa':>7}  verdict", file=sys.stderr)
    for item, stats in results.items():
        kappa = stats["kappa"]
        if stats["n"] == 0:
            verdict = "NO LABELS"
        elif kappa is None:
            verdict = "UNDEFINED (no variance — not a pass)"
        elif kappa >= KAPPA_FLOOR:
            verdict = "ok"
        else:
            verdict = f"DROP (< {KAPPA_FLOOR})"
        kappa_text = f"{kappa:.3f}" if kappa is not None else "  n/a"
        raw_text = f"{stats['raw_agreement']:.3f}" if stats["raw_agreement"] is not None else "  n/a"
        print(f"[validate] {item:<28} {stats['n']:>5} {raw_text:>7} {kappa_text:>7}  {verdict}", file=sys.stderr)

    return results
