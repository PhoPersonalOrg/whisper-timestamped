"""Backend-agnostic token-loop detection.

Post-hoc loop detection (:func:`find_token_loop`) scans a finished token
sequence for repeated n-gram patterns.  It is used by *every* backend --
the CTranslate2 repair/blocking strategies in
:mod:`crisperwhisper.hallucination`, the Transformers backend's
``generate_with_repair``, and the temperature fallback -- so it lives in
this dependency-free module rather than in ``hallucination.py``, which
imports ``ctranslate2`` at module level (issue #57).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DEFAULT_REPAIR_THRESHOLDS: dict[int, int] = {
    1: 8, 2: 8, 3: 4, 4: 3, 5: 3, **{n: 3 for n in range(6, 25)},
}
"""Per-ngram-size repetition thresholds for ``generate_with_repair``.

Keys are n-gram sizes, values are the number of consecutive repetitions
that trigger a repair.  Larger n-grams use lower thresholds because
long repeated phrases are almost never genuine speech.

Sizes extend to 24 because whole-sentence loops are real: an observed
failure repeated an 11-token sentence ("I have to think about what I'm
thinking about.") 21 times, invisible to a scan capped at 5-grams.
Scanning up to 24 covers such units with ~2x headroom at negligible
cost (the scan is a cheap post-hoc pass over <=448-token sequences).
"""


def find_token_loop(
    ids: list[int],
    min_ngram: int = 1,
    max_ngram: int = 5,
    reps: int | dict[int, int] = 8,
) -> tuple[int, tuple[int, ...]] | None:
    """Find the earliest position where consecutive n-grams repeat.

    Parameters
    ----------
    ids : list[int]
        Token ID sequence to scan.
    min_ngram, max_ngram : int
        Range of n-gram sizes to check when *reps* is an ``int``.
        Ignored when *reps* is a ``dict`` — the dict's keys alone define
        the scanned sizes, so callers with a per-size threshold table
        never silently drop its larger entries behind a stale cap.
    reps : int or dict[int, int]
        If an ``int``, the same threshold is used for all n-gram sizes
        in ``[min_ngram, max_ngram]``.  If a ``dict``, maps
        ``{ngram_size: threshold}`` and exactly those sizes are scanned.

    Returns ``(loop_start_index, ngram_tuple)`` or ``None``.
    """
    thresholds: dict[int, int]
    if isinstance(reps, int):
        thresholds = {n: reps for n in range(min_ngram, max_ngram + 1)}
    else:
        thresholds = reps

    for i in range(len(ids)):
        for n, needed_reps in thresholds.items():
            needed = n * needed_reps
            if i + needed > len(ids):
                continue
            gram = tuple(ids[i : i + n])
            if all(
                tuple(ids[i + r * n : i + (r + 1) * n]) == gram
                for r in range(1, needed_reps)
            ):
                return i, gram
    return None


def _block_thresholds(
    detect_reps: int | dict[int, int], max_ngram: int = 5,
) -> dict[int, int]:
    """Blocking thresholds one repetition below the detection thresholds.

    ``_compute_bans`` bans the token that would start copy ``r + 1`` once
    ``r`` consecutive copies exist, so blocking at ``detect - 1`` keeps
    every consecutive repetition strictly below the detection threshold —
    after a blocked continuation the loop detector can never fire again,
    no matter how the decoder mutates the looping unit.
    """
    if isinstance(detect_reps, int):
        return {n: max(1, detect_reps - 1) for n in range(1, max_ngram + 1)}
    return {n: max(1, r - 1) for n, r in detect_reps.items()}


DEGENERATE_TAIL_RATIO = 0.5
"""Minimum unique-5-gram ratio for a blocked continuation to be kept.

Exact-match blocking stops exact loops, but a semantically stuck decoder
then emits NEAR-copies that mutate by a token or two each repetition —
invisible to exact n-gram detection, still garbage.  Genuine speech sits
near 1.0 on this metric; observed degenerate tails sit near 0.15.
"""


def _unique_ngram_ratio(ids: list[int], n: int = 5) -> float:
    """Fraction of distinct n-grams among all n-grams of ``ids``."""
    if len(ids) < 2 * n:
        return 1.0
    grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / len(grams)


def _final_trim(
    generated: list[int],
    detect_reps: int | dict[int, int],
    keep_reps: int,
) -> list[int]:
    """Last-resort truncation when the repair budget is exhausted.

    Should be unreachable when repair attempt 2+ continues under
    persistent n-gram blocking, but kept as a guarantee: a loop must
    never ship.  Cut the output at loop onset + ``keep_reps`` copies and
    let the transcript end there (equivalent to a budget-exhausted
    decode, which downstream longform handling already tolerates).
    """
    hit = find_token_loop(generated, reps=detect_reps)
    if hit is None:
        return generated
    loop_start, gram = hit
    keep_end = loop_start + len(gram) * keep_reps
    logger.warning(
        "Loop persisted after repair budget (%d-gram at pos %d); "
        "truncating output from %d to %d tokens",
        len(gram), loop_start, len(generated), keep_end,
    )
    return generated[:keep_end]
