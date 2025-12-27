"""Utilities for parsing metric references used across training/ops scheduling.

This module is intentionally dependency-light so it can be imported in unit tests
without requiring heavyweight ML deps (e.g. torch).
"""

from __future__ import annotations


def parse_metric_ref(metric: str, *, default_phase: str = "val") -> tuple[str, str]:
    """Parse a metric reference into (phase, metric_name).

    Supported forms:
    - ``val/loss`` (preferred explicit form)
    - ``train/loss``
    - ``val_loss`` / ``train_loss`` (legacy shorthand)
    - ``val.loss`` / ``train.loss`` (legacy shorthand)

    If the phase is missing, defaults to ``default_phase``.
    """
    raw = (metric or "").strip()
    if not raw:
        return default_phase, "loss"

    lowered = raw.lower()

    for sep in ("/", "."):
        if sep in lowered:
            phase, name = lowered.split(sep, 1)
            phase = phase.strip() or default_phase
            name = name.strip() or "loss"
            return phase, name

    for prefix in ("train_", "val_"):
        if lowered.startswith(prefix):
            phase = prefix[:-1]  # drop trailing "_"
            name = lowered[len(prefix) :].strip() or "loss"
            return phase, name

    return default_phase, lowered

