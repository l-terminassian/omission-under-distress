# Self-Over-Other

Does a language model's honesty degrade as the user appears more distressed?

The experiment holds a risky decision fixed and varies three things independently:
how vulnerable the user sounds, whose decision it is, and who is said to have
recommended it. Each factor controls exactly one sentence of a three-sentence
prompt, so any behavioural difference is attributable to that factor rather than
to incidental wording. Responses are scored against a rubric whose key item —
did the reply state this scenario's specific risk? — is checkable because the
risk is written into the scenario before the run.

The analysis plan is pre-registered in `soo/config.py`: primary outcome, primary
and secondary interactions, the reliability floor for dropping a rubric item, and
the white-box outcome, layer band and predicted sign. Analyses not fixed there
are reported as exploratory.

## Install

```bash
uv sync --extra dev
```

Python 3.12 or 3.13. Add `--extra whitebox` for the activation probe; see below.

## API keys

Copy the template and fill in the providers you plan to use:

```bash
cp .env.example .env
```

`.env` is gitignored. Keys are read from the environment at import time and never
written to any output file. Anthropic and OpenAI keys are read directly; Bedrock
uses the standard AWS credential chain, so either set `AWS_PROFILE` in `.env` or
export `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` yourself.

You do not need all three. `soo run --models` restricts the run to whichever
providers you have configured.

## Reproduce

```bash
uv run soo run --dry-run     # full grid against a stub client, no API calls
uv run soo run               # generate responses  (writes outputs/responses.jsonl)
uv run soo judge             # score them against the rubric
uv run soo crossjudge        # re-score a subsample with a second vendor
uv run soo manipcheck        # verify the cues are recoverable from the prompt
uv run soo analyse           # pre-registered analysis  -> outputs/analysis.md
uv run soo robustness        # mixed effects, paired, leave-one-domain, judges
uv run soo figures           # diagnostic figures
```

Start with `--dry-run`, which exercises the whole pipeline for free. Add
`--limit 2` to any command for a two-scenario smoke test against real providers.

Every record carries a deterministic id and is flushed as it completes, so an
interrupted run resumes where it stopped — re-run the same command.

## Judge validation

The rubric is scored by a model, so it is checked against hand labels before any
claim rests on it:

```bash
uv run soo validate-export   # blind CSV: no condition labels, no judge scores
# fill in the rubric columns by hand
uv run soo validate-score    # judge-vs-human Cohen's kappa per item
```

Items scoring below `KAPPA_FLOOR` are dropped from headline claims. That
threshold is set in `soo/config.py` before the numbers exist.
`validate-extend` appends a second labelling batch drawn from the
neutral-vs-distressed contrast, where the labels buy the most power.

## White-box probe

Correlates a model's internal self-vs-other representational distance with its
own behaviour on the same prompt. Requires the optional extra and an accelerator
with room for the probe model in fp16 — about 15 GB for the 7B default.

```bash
uv sync --extra whitebox
uv run soo whitebox
```

The probe writes distances per layer; `soo judge` then scores the generated
responses, and the correlation runs against the pre-registered outcome and layer
band. Set `SOO_WHITEBOX_SAVE_STATES=1` to retain raw hidden states, which are
needed to repeat the correlation on mean-centred vectors.

## Layout

```
soo/
  config.py           constants and the pre-registered analysis plan
  conditions.py       prompt construction; one sentence per factor
  scenarios.py        scenario schema and validation
  rubric.py           rubric items, judge prompt, JSON schema
  clients.py          async Anthropic / OpenAI / Bedrock, retries, stub client
  run_behavioural.py  generate the response grid
  judge.py            score responses; manipulation check
  validate.py         export labelling CSVs; judge-vs-human kappa
  repudiation.py      detect replies that reject the attribution premise
  whitebox.py         activation probe and its pre-registered correlation
  analyse.py          pre-registered analysis -> analysis.md
  robustness.py       robustness checks behind the headline claims
  figures.py          diagnostic figures
  register.py         formal-vs-casual rewrite check
  cli.py              subcommands
data/scenarios.jsonl  30 scenarios, 6 domains, each with a written key risk
scripts/              figure generation for the write-up
tests/                rendering invariants, scenario schema, resume logic
```

## Tests

```bash
uv run pytest
uv run ruff check .
```

The rendering tests are the important ones: they assert that the request
sentence is byte-identical across vulnerability levels within an attribution. If
that invariant breaks, every downstream number becomes uninterpretable.
