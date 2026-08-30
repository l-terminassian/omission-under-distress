"""
soo/judge.py — rubric scoring, manipulation check, cross-judge sensitivity.

Three jobs:

1. **score** — every response, both turns, against the rubric.
2. **manipulation check** — can a reader recover the vulnerability cue from the
   prompt alone? Run this *first*: if the cues are not recoverable, a null
   result carries no information and the money should not be spent.
3. **cross-judge** — re-score a random subsample with a different family, so the
   "our judge shares a vendor with one model under test" objection has a number
   attached rather than a shrug.

The judge runs with thinking disabled and structured outputs: cheaper, lower
variance, and the rubric JSON is schema-guaranteed rather than parsed on a hope.
"""
from __future__ import annotations

import asyncio
import json
import random
import sys

from .clients import chat
from .conditions import grid
from .config import (
    CROSS_JUDGE_FRACTION,
    CROSS_JUDGE_MODEL,
    CROSS_JUDGE_PROVIDER,
    CROSS_SCORES_PATH,
    JUDGE_MODEL,
    JUDGE_PROVIDER,
    MANIPCHECK_PATH,
    MAX_CONCURRENCY,
    MAX_TOKENS_JUDGE,
    RESPONSES_PATH,
    SCORES_PATH,
    VALIDATION_SEED,
)
from .rubric import (
    JUDGE_SYSTEM,
    MANIPCHECK_SCHEMA,
    MANIPCHECK_SYSTEM,
    build_judge_prompt,
    build_manipcheck_prompt,
    parse_rubric_json,
    rubric_schema,
)
from .scenarios import load_scenarios
from .store import JsonlWriter, completed_ids, read_jsonl, record_id


def _judge_target(dry_run: bool, cross: bool) -> tuple[str, str]:
    if dry_run:
        return "stub", "stub-judge"
    if cross:
        return CROSS_JUDGE_PROVIDER, CROSS_JUDGE_MODEL
    return JUDGE_PROVIDER, JUDGE_MODEL

async def _score_one(
    response: dict,
    scenario: dict,
    turn: int,
    provider: str,
    model_id: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    rid = record_id(response["record_id"], turn, model_id)
    prompt = build_judge_prompt(response, scenario, turn)
    schema = rubric_schema(turn)

    result = await chat(
        provider,
        model_id,
        [{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS_JUDGE,
        system=JUDGE_SYSTEM,
        json_schema=schema,
        disable_thinking=True,
        semaphore=semaphore,
        label=f"judge/{response['record_id'][:8]}/t{turn}",
    )

    scored = parse_rubric_json(result["text"], turn) if not result["error"] else None
    return {
        "record_id": rid,
        "response_record_id": response["record_id"],
        "judge_model": model_id,
        "turn": turn,
        "model_key": response["model_key"],
        "family": response.get("family"),
        "scenario_id": response["scenario_id"],
        "domain": response["domain"],
        "attribution": response["attribution"],
        "vulnerability": response["vulnerability"],
        "advice_source": response.get("advice_source"),
        "sample": response["sample"],
        "stop_reason": response.get(f"stop_reason_turn{turn}", ""),
        "scores": scored,
        "error": result["error"] or (None if scored else "unparseable judge output"),
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
    }

async def score(dry_run: bool = False, cross: bool = False, max_concurrency: int = MAX_CONCURRENCY) -> dict:
    """Score responses.jsonl against the rubric."""
    responses = read_jsonl(RESPONSES_PATH)
    if not responses:
        raise RuntimeError(f"no responses found at {RESPONSES_PATH}; run `soo run` first")

    scenarios = {s["id"]: s for s in load_scenarios()}
    provider, model_id = _judge_target(dry_run, cross)
    out_path = CROSS_SCORES_PATH if cross else SCORES_PATH

    usable = [r for r in responses if not r.get("error") and r.get("response_turn1", "").strip()]

    if cross:
        # A random subsample is enough to estimate agreement, and re-judging
        # everything with a second family would roughly double judging cost.
        rng = random.Random(VALIDATION_SEED)
        usable = rng.sample(usable, max(1, int(len(usable) * CROSS_JUDGE_FRACTION)))

    done = completed_ids(out_path)
    jobs: list[tuple[dict, int]] = []
    for response in usable:
        for turn in (1, 2):
            if turn == 2 and not response.get("response_turn2", "").strip():
                continue
            if record_id(response["record_id"], turn, model_id) not in done:
                jobs.append((response, turn))

    print(
        f"[judge] {'cross-judge' if cross else 'judge'} {model_id}: "
        f"{len(jobs)} to score ({len(done)} already done)",
        file=sys.stderr,
    )
    if not jobs:
        return {"written": 0, "skipped": len(done)}

    semaphore = asyncio.Semaphore(max_concurrency)
    written, failures = 0, 0
    totals = {"input_tokens": 0, "output_tokens": 0}

    with JsonlWriter(out_path) as writer:
        pending = [
            asyncio.create_task(_score_one(r, scenarios[r["scenario_id"]], t, provider, model_id, semaphore))
            for r, t in jobs
        ]
        for finished in asyncio.as_completed(pending):
            row = await finished
            writer.write(row)
            written += 1
            failures += 1 if row["error"] else 0
            totals["input_tokens"] += row["input_tokens"]
            totals["output_tokens"] += row["output_tokens"]
            if written % 50 == 0 or written == len(jobs):
                print(f"[judge] {written}/{len(jobs)}", file=sys.stderr)

    print(f"[judge] wrote {written} scores to {out_path} ({failures} failed)", file=sys.stderr)
    return {"written": written, "failures": failures, **totals}

async def manipulation_check(dry_run: bool = False, limit: int | None = None) -> dict:
    """Recover the vulnerability level from the prompt alone.

    Reported before any results. If accuracy is below MANIPCHECK_FLOOR the cues
    are too subtle for a null to be informative and the run should stop.
    """
    scenarios = load_scenarios(limit=limit)
    cells = grid(scenarios)
    provider, model_id = _judge_target(dry_run, cross=False)

    done = completed_ids(MANIPCHECK_PATH)
    jobs = [
        c
        for c in cells
        if record_id("manipcheck", c["scenario_id"], c["attribution"], c["vulnerability"],
                  c["advice_source"], model_id) not in done
    ]
    print(f"[manipcheck] {len(jobs)} prompts to classify ({len(done)} already done)", file=sys.stderr)

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _one(cell: dict) -> dict:
        rid = record_id("manipcheck", cell["scenario_id"], cell["attribution"], cell["vulnerability"],
                        cell["advice_source"], model_id)
        result = await chat(
            provider,
            model_id,
            [{"role": "user", "content": build_manipcheck_prompt(cell["prompt"])}],
            max_tokens=256,
            system=MANIPCHECK_SYSTEM,
            json_schema=MANIPCHECK_SCHEMA,
            disable_thinking=True,
            semaphore=semaphore,
            label=f"manipcheck/{cell['scenario_id']}",
        )
        guess = None
        if not result["error"]:
            try:
                guess = json.loads(result["text"]).get("vulnerability")
            except Exception:  # noqa: BLE001 - a bad guess is data, not a crash
                guess = None
        return {
            "record_id": rid,
            "scenario_id": cell["scenario_id"],
            "attribution": cell["attribution"],
            "vulnerability": cell["vulnerability"],
            "advice_source": cell["advice_source"],
            "guess": guess,
            "correct": int(guess == cell["vulnerability"]) if guess else 0,
            "error": result["error"],
        }

    written = 0
    if jobs:
        with JsonlWriter(MANIPCHECK_PATH) as writer:
            for finished in asyncio.as_completed([asyncio.create_task(_one(c)) for c in jobs]):
                writer.write(await finished)
                written += 1

    rows = read_jsonl(MANIPCHECK_PATH)
    scored = [r for r in rows if r.get("guess")]
    accuracy = sum(r["correct"] for r in scored) / len(scored) if scored else 0.0

    by_attribution: dict[str, list[int]] = {}
    for row in scored:
        by_attribution.setdefault(row["attribution"], []).append(row["correct"])

    print(f"[manipcheck] 4-way recovery accuracy = {accuracy:.3f} over {len(scored)} prompts", file=sys.stderr)
    for attribution, hits in sorted(by_attribution.items()):
        print(f"[manipcheck]   {attribution}: {sum(hits) / len(hits):.3f} (n={len(hits)})", file=sys.stderr)

    return {"written": written, "accuracy": accuracy, "n": len(scored)}
