"""
soo/figures.py — the three figures the write-up needs.

Design decisions, so they are not re-litigated by taste later:

* **Palette.** Categorical slots 1-3 of the reference palette (blue / orange /
  aqua), used in fixed order, never cycled. That triple is documented as
  clearing the all-pairs colour-vision gates in both modes. Aqua sits below 3:1
  contrast on a light surface, so the relief rule applies: every bar carries a
  visible value label, and analysis.md ships the same numbers as a table.
* **Text never wears the series colour.** Values and labels stay in ink; the
  coloured mark beside them carries identity.
* **One axis, always.** No dual-scale plots anywhere.
* **Figure 2 is the headline.** The interaction coefficient with its confidence
  interval *is* the result, so it gets its own figure rather than being buried
  in a table.
"""
from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import numpy as np

from .analyse import build_frame, clustered_bootstrap_rate, fit_interaction
from .config import (
    ADVICE_SOURCES,
    ATTRIBUTIONS,
    FIGURES_DIR,
    PRIMARY_OUTCOME,
    PRIMARY_TURN,
    VULNERABILITIES,
    WHITEBOX_PATH,
)


# Reference palette, categorical slots 1-3 (light mode), in fixed order.
# Slot 1/2 carry the PRIMARY axis (advice source); slot 3 is unused here because
# the primary figure has only two series.
SERIES_COLORS = {"model_advised": "#2a78d6", "other_advised": "#eb6834",
                 "first_person": "#2a78d6", "friend": "#eb6834"}
SERIES_LABELS = {
    "model_advised": "the model gave the advice",
    "other_advised": "another AI gave the advice",
    "first_person": "their own decision",
    "friend": "a friend's decision",
}
VULN_LABELS = {"neutral": "neutral", "lonely": "lonely", "bereaved": "bereaved", "distressed": "in a dark place"}

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"

def _style(ax) -> None:
    """Recessive axes and grid; the data carries the emphasis."""
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK_MUTED)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=3, width=0.8)
    ax.grid(axis="y", color=INK_MUTED, alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)

def fig_rates(frame, outcome: str = PRIMARY_OUTCOME, series: str = "advice_source",
              filename: str = "fig1_rates.png") -> None:
    """Rate by vulnerability, grouped by `series`, faceted by model."""


    models = sorted(frame["model_key"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.4), sharey=True, facecolor=SURFACE)
    if len(models) == 1:
        axes = [axes]

    levels = ADVICE_SOURCES if series == "advice_source" else ATTRIBUTIONS
    width = 0.8 / len(levels)
    positions = np.arange(len(VULNERABILITIES))

    for ax, model_key in zip(axes, models, strict=True):
        _style(ax)
        subset = frame[frame["model_key"] == model_key]
        for index, level in enumerate(levels):
            rates, errors = [], []
            for vulnerability in VULNERABILITIES:
                cell = subset[(subset[series] == level) & (subset["vulnerability"] == vulnerability)]
                if cell.empty:
                    rates.append(np.nan)
                    errors.append(0.0)
                    continue
                point, low, high = clustered_bootstrap_rate(cell, outcome, iterations=600)
                rates.append(point)
                errors.append(max(point - low, high - point) if not np.isnan(low) else 0.0)

            offset = (index - (len(levels) - 1) / 2) * width
            bars = ax.bar(
                positions + offset,
                rates,
                width * 0.92,  # a 2px-equivalent surface gap between adjacent bars
                color=SERIES_COLORS[level],
                label=SERIES_LABELS[level],
                zorder=2,
            )
            ax.errorbar(
                positions + offset, rates, yerr=errors, fmt="none",
                ecolor=INK_SECONDARY, elinewidth=1.2, capsize=3, zorder=3,
            )
            # Relief rule: visible labels, in ink rather than the series colour.
            # Sat above the bar top (clear of the error bar) rather than pinned to
            # the baseline, so they stay legible whatever the rate.
            for bar, rate, error in zip(bars, rates, errors, strict=True):
                if not np.isnan(rate):
                    ax.text(bar.get_x() + bar.get_width() / 2, rate + error + 0.03, f"{rate:.2f}",
                            ha="center", va="bottom", fontsize=7, color=INK_PRIMARY, zorder=4)

        ax.set_xticks(positions)
        ax.set_xticklabels([VULN_LABELS[v] for v in VULNERABILITIES], fontsize=9)
        ax.set_title(model_key, fontsize=11, color=INK_PRIMARY, pad=10)
        ax.set_ylim(0, 1.18)  # headroom for the value labels

    axes[0].set_ylabel("share of replies that stated the key risk", fontsize=10, color=INK_SECONDARY)
    # Legend below the plots: inside the axes it collides with bars or labels
    # depending on the data, which is exactly the kind of fault that only shows
    # up once real numbers arrive.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, 0.0), labelcolor=INK_SECONDARY)
    fig.suptitle(
        f"Did the model warn? `{outcome}` at turn {PRIMARY_TURN}, by how vulnerable the decision-maker looks",
        fontsize=12, color=INK_PRIMARY, y=0.99,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"[figures] wrote {path}", file=sys.stderr)

def fig_interaction(results: dict, filename: str = "fig2_interaction.png") -> None:
    """The headline: interaction coefficient per model, with 95% CI."""

    entries = []
    for model_key, result in results.items():
        if not result.get("ok"):
            continue
        term = result["terms"].get(f"vulnerable:{result.get('moderator', 'model_advised')}")
        if term:
            label = "pooled (secondary)" if model_key == "_pooled" else model_key
            entries.append((label, term))
    if not entries:
        print("[figures] no interaction estimates to plot", file=sys.stderr)
        return

    moderator = next((r.get("moderator") for r in results.values() if r.get("ok")), "model_advised")
    titles = {
        "model_advised": "Is the model less honest about risk when told the advice was its own?",
        "first_person": "Is the honesty cost of vulnerability larger when it's the user's own decision?",
    }
    subtitles = {
        "model_advised": "negative = larger honesty cost when the model is told it gave the advice",
        "first_person": "negative = larger honesty cost when the vulnerable person is the one asking",
    }

    fig, ax = plt.subplots(figsize=(7.8, 0.62 * len(entries) + 1.9), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=INK_MUTED, alpha=0.18, linewidth=0.8)

    ys = np.arange(len(entries))
    for y, (_label, term) in zip(ys, entries, strict=True):
        ax.plot([term["ci_low"], term["ci_high"]], [y, y], color=INK_SECONDARY, linewidth=2, zorder=2)
        ax.plot([term["coef"]], [y], "o", markersize=9, color=SERIES_COLORS["first_person"],
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ax.text(term["ci_high"], y, f"   {term['coef']:+.2f}  (p={term['p']:.3f})",
                fontsize=8.5, color=INK_PRIMARY, va="center", ha="left")

    ax.axvline(0, color=INK_MUTED, linewidth=1.2, linestyle="--", zorder=1)
    span = max(t["ci_high"] for _, t in entries) - min(t["ci_low"] for _, t in entries)
    ax.set_xlim(min(t["ci_low"] for _, t in entries) - 0.05 * span,
                max(t["ci_high"] for _, t in entries) + 0.55 * span)
    ax.set_yticks(ys)
    ax.set_yticklabels([label for label, _ in entries], fontsize=10)
    ax.set_title(titles.get(moderator, "Interaction estimates"), fontsize=12, color=INK_PRIMARY, pad=12)
    ax.set_xlabel(
        f"interaction coefficient (log-odds)   —   {subtitles.get(moderator, '')}",
        fontsize=9, color=INK_SECONDARY,
    )
    ax.margins(y=0.25)
    fig.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"[figures] wrote {path}", file=sys.stderr)

def fig_whitebox() -> None:
    """Per-layer self-other distance coefficient. Single series, so no legend."""

    path_in = WHITEBOX_PATH.parent / "whitebox_correlation.json"
    if not path_in.is_file():
        print("[figures] no whitebox correlation yet; skipping fig3", file=sys.stderr)
        return
    payload = json.loads(path_in.read_text())
    layers = payload.get("layers", [])
    if not layers:
        return

    fig, ax = plt.subplots(figsize=(7.4, 4.0), facecolor=SURFACE)
    _style(ax)
    xs = [entry["layer"] for entry in layers]
    ys = [entry["coef_per_sd"] for entry in layers]
    ax.plot(xs, ys, linewidth=2, color=SERIES_COLORS["first_person"], zorder=2)

    best = payload.get("best")
    if best:
        ax.plot([best["layer"]], [best["coef_per_sd"]], "o", markersize=9,
                color=SERIES_COLORS["friend"],  # slot 2 (orange), marking the most predictive layer
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ax.annotate(
            f"layer {best['layer']}  (p={best['p']:.3f}, uncorrected)",
            xy=(best["layer"], best["coef_per_sd"]), xytext=(8, 10), textcoords="offset points",
            fontsize=9, color=INK_PRIMARY,
        )

    ax.axhline(0, color=INK_MUTED, linewidth=1.2, linestyle="--", zorder=1)
    ax.set_xlabel("layer", fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel("log-odds per SD of self-other distance", fontsize=10, color=INK_SECONDARY)
    ax.set_title("Does self-vs-user representational distance predict honesty?", fontsize=12,
                 color=INK_PRIMARY, pad=12)
    fig.tight_layout()
    out = FIGURES_DIR / "fig3_whitebox_layers.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"[figures] wrote {out}", file=sys.stderr)

def make_figures() -> None:

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    frame, _ = build_frame()
    primary = frame[frame["turn"] == PRIMARY_TURN]

    fig_rates(primary, series="advice_source", filename="fig1_rates_by_source.png")
    fig_rates(primary, series="attribution", filename="fig1b_rates_by_attribution.png")

    results = {key: fit_interaction(primary[primary["model_key"] == key], PRIMARY_OUTCOME,
                                    moderator="model_advised")
               for key in sorted(primary["model_key"].unique())}
    results["_pooled"] = fit_interaction(primary, PRIMARY_OUTCOME, moderator="model_advised")
    fig_interaction(results, filename="fig2_interaction_primary.png")

    secondary = {key: fit_interaction(primary[primary["model_key"] == key], PRIMARY_OUTCOME,
                                      moderator="first_person")
                 for key in sorted(primary["model_key"].unique())}
    secondary["_pooled"] = fit_interaction(primary, PRIMARY_OUTCOME, moderator="first_person")
    fig_interaction(secondary, filename="fig2b_interaction_secondary.png")

    fig_whitebox()
