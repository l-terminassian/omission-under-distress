"""
soo/run_behavioural.py — generate the response grid.

One unit of work is (model, scenario, attribution, vulnerability, sample). Each
unit is two sequential calls — the request, then the same conversation plus a
pushback line — because turn 2 has to see turn 1's actual reply. Units run
concurrently under a single semaphore.

Cost and truncation are tracked per record. Truncation matters more than it
looks: a reply cut off before it reached its risk warning would score
``named_risk = 0`` for a mechanical reason and bias the primary outcome
downward, so `stop_reason` is kept on every row and truncated rows are excluded
from the primary analysis rather than counted as failures to warn.
"""
from __future__ import annotations

import asyncio
import sys
import time

from .clients import chat, resolve_bedrock_model
from .conditions import SYSTEM_PROMPT, control_grid, dose_grid, grid
from .config import (
    EXTRA_MODELS,
    MAX_CONCURRENCY,
    MAX_TOKENS_RESPONSE,
    MODELS_UNDER_TEST,
    N_SAMPLES,
    RESPONSES_PATH,
)
from .scenarios import load_scenarios
from .store import JsonlWriter, completed_ids, record_id


def resolve_models(dry_run: bool, only: list[str] | None = None,
                   roster: list[dict] | None = None) -> list[dict]:
    """Materialise the model roster, resolving the Bedrock family against the API.

    In dry-run everything becomes the stub provider so the pipeline can be
    exercised with no credentials and no spend.
    """
    models: list[dict] = []
    for spec in (roster if roster is not None else MODELS_UNDER_TEST):
        if only and spec["key"] not in only:
            continue
        entry = dict(spec)
        if dry_run:
            entry["provider"] = "stub"
            entry["model_id"] = entry["model_id"] or "stub-model"
        elif entry["provider"] == "bedrock" and not entry["model_id"]:
            entry["model_id"] = resolve_bedrock_model()
        models.append(entry)
    return models


def _unit_id(model: dict, cell: dict, sample: int) -> str:
    """Identity of one grid cell for a given model.

    The provider is part of the id on purpose. Without it, `--dry-run` stub
    records would carry the same ids as real ones, and a subsequent real run
    would skip them as "already done" — silently shipping stub text into the
    analysis. Including the provider makes that collision impossible.
    """
    return record_id(
        model["provider"], model["key"], cell["scenario_id"], cell["attribution"], cell["vulnerability"],
        cell["advice_source"], sample
    )


async def _run_unit(model: dict, cell: dict, sample: int, semaphore: asyncio.Semaphore) -> dict:
    """Both turns for one grid cell, as one record."""
    rid = _unit_id(model, cell, sample)
    label = (f"{model['key']}/{cell['scenario_id']}/{cell['attribution']}"
             f"/{cell['vulnerability']}/{cell['advice_source']}/s{sample}")

    # prior_turns is empty except in the structural positive-control arm,
    # where it carries a genuine assistant turn recommending the plan.
    messages = list(cell.get("prior_turns", [])) + [{"role": "user", "content": cell["prompt"]}]
    first = await chat(
        model["provider"],
        model["model_id"],
        messages,
        max_tokens=MAX_TOKENS_RESPONSE,
        system=SYSTEM_PROMPT,
        semaphore=semaphore,
        label=f"{label}/t1",
    )

    record = {
        "record_id": rid,
        "model_key": model["key"],
        "model_id": model["model_id"],
        "family": model["family"],
        "sample": sample,
        **{k: cell[k] for k in
           ("scenario_id", "domain", "attribution", "vulnerability", "advice_source", "prompt", "pushback")},
        "response_turn1": first["text"],
        "stop_reason_turn1": first["stop_reason"],
        "error": first["error"],
        "input_tokens": first["input_tokens"],
        "output_tokens": first["output_tokens"],
    }

    # Turn 2 only makes sense if turn 1 produced something to push back against.
    if first["error"] or not first["text"].strip():
        record.update({"response_turn2": "", "stop_reason_turn2": "skipped"})
        return record

    messages_turn2 = messages + [
        {"role": "assistant", "content": first["text"]},
        {"role": "user", "content": cell["pushback"]},
    ]
    second = await chat(
        model["provider"],
        model["model_id"],
        messages_turn2,
        max_tokens=MAX_TOKENS_RESPONSE,
        system=SYSTEM_PROMPT,
        semaphore=semaphore,
        label=f"{label}/t2",
    )

    record.update(
        {
            "response_turn2": second["text"],
            "stop_reason_turn2": second["stop_reason"],
            "input_tokens": record["input_tokens"] + second["input_tokens"],
            "output_tokens": record["output_tokens"] + second["output_tokens"],
        }
    )
    if second["error"] and not record["error"]:
        record["error"] = second["error"]
    return record


async def run(
    dry_run: bool = False,
    limit: int | None = None,
    only_models: list[str] | None = None,
    samples: int = N_SAMPLES,
    max_concurrency: int = MAX_CONCURRENCY,
    control: bool = False,
    dose: bool = False,
) -> dict:
    """Fill the grid, skipping records already completed successfully.

    With `control=True` runs only the structural positive-control arm instead of
    the main factorial. Both write to the same responses.jsonl; the
    `advice_source` field keeps them apart.
    """
    scenarios = load_scenarios(limit=limit)
    if dose:
        cells = dose_grid(scenarios)
    elif control:
        cells = control_grid(scenarios)
    else:
        cells = grid(scenarios)
    models = resolve_models(dry_run, only_models, roster=EXTRA_MODELS if dose else None)

    done = completed_ids(RESPONSES_PATH)
    units: list[tuple[dict, dict, int]] = []
    for model in models:
        for cell in cells:
            for sample in range(samples):
                if _unit_id(model, cell, sample) not in done:
                    units.append((model, cell, sample))

    total = len(models) * len(cells) * samples
    print(
        f"[run] {len(models)} model(s) x {len(cells)} cells x {samples} samples = {total} records; "
        f"{len(done)} already done, {len(units)} to run",
        file=sys.stderr,
    )
    if not units:
        return {"written": 0, "skipped": len(done), "input_tokens": 0, "output_tokens": 0}

    semaphore = asyncio.Semaphore(max_concurrency)
    started = time.time()
    written = 0
    totals = {"input_tokens": 0, "output_tokens": 0, "errors": 0, "truncated": 0}

    with JsonlWriter(RESPONSES_PATH) as writer:
        pending = [asyncio.create_task(_run_unit(m, c, s, semaphore)) for m, c, s in units]
        for finished in asyncio.as_completed(pending):
            record = await finished
            writer.write(record)
            written += 1
            totals["input_tokens"] += record["input_tokens"]
            totals["output_tokens"] += record["output_tokens"]
            if record.get("error"):
                totals["errors"] += 1
            if "max_tokens" in (record.get("stop_reason_turn1", ""), record.get("stop_reason_turn2", "")):
                totals["truncated"] += 1
            if written % 25 == 0 or written == len(units):
                rate = written / max(time.time() - started, 1e-6)
                print(f"[run] {written}/{len(units)} ({rate:.1f}/s)", file=sys.stderr)

    print(
        f"[run] wrote {written} records to {RESPONSES_PATH} "
        f"({totals['errors']} errors, {totals['truncated']} truncated, "
        f"{totals['input_tokens']} in / {totals['output_tokens']} out tokens)",
        file=sys.stderr,
    )
    return {"written": written, "skipped": len(done), **totals}
