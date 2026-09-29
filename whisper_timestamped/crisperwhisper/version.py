"""CrisperWhisper model-version detection.

Distinguishes the legacy v1 model (``nyrahealth/CrisperWhisper`` -- a plain
Whisper model with a changed tokenizer) from v2
(``nyralabs/CrisperWhisper2.0_<size>`` and its derivatives) by checking the
tokenizer files for v2 marker tokens.

Both versions run on :class:`crisperwhisper.transformers_engine.TransformersEngine`
(and v2 additionally on the CTranslate2 backend); the version only controls the
decoder prompt (plain prefix vs. verbatim/intended tags) and which features are
available, so detection is kept deliberately lightweight (no model load).
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_V2_MARKER = "[verbatim_1]"


def _check_local_dir(path: Path) -> int | None:
    """Check a local model directory for v2 marker tokens.

    Looks in ``added_tokens.json`` (dict or list), ``vocab.json``,
    and ``tokenizer.json`` -- whichever exists first.
    """
    import json

    added_path = path / "added_tokens.json"
    if added_path.exists():
        data = json.loads(added_path.read_text())
        if isinstance(data, dict):
            return 2 if _V2_MARKER in data else 1
        if isinstance(data, list):
            strs = {t.get("content", "") if isinstance(t, dict) else str(t) for t in data}
            return 2 if _V2_MARKER in strs else 1

    vocab_path = path / "vocab.json"
    if vocab_path.exists():
        v = json.loads(vocab_path.read_text())
        return 2 if _V2_MARKER in v else 1

    tok_path = path / "tokenizer.json"
    if tok_path.exists():
        tok = json.loads(tok_path.read_text())
        vocab = tok.get("model", {}).get("vocab", {})
        if _V2_MARKER in vocab:
            return 2
        added = tok.get("added_tokens", [])
        strs = {t.get("content", "") if isinstance(t, dict) else str(t) for t in added}
        return 2 if _V2_MARKER in strs else 1

    return None


def _is_official_v2_id(model_name_or_path: str) -> bool:
    """True for known CrisperWhisper 2.0 Hub ids (no Hub I/O needed)."""
    from whisper_timestamped.crisperwhisper.model import OFFICIAL_MODELS

    if model_name_or_path in OFFICIAL_MODELS.values():
        return True
    # Accept future size tags under the same org/prefix.
    name = model_name_or_path.split("/")[-1]
    return model_name_or_path.startswith("nyralabs/") and name.startswith(
        "CrisperWhisper2.0_"
    )


def detect_model_version(model_name_or_path: str) -> int:
    """Detect CrisperWhisper model version without loading the full model.

    Checks tokenizer files for v2 marker tokens. Returns 1 or 2.
    """
    path = Path(model_name_or_path)

    if path.is_dir():
        result = _check_local_dir(path)
        if result is not None:
            logger.info("Detected CrisperWhisper v%d from local dir", result)
            return result

    if _is_official_v2_id(model_name_or_path):
        logger.info("Detected CrisperWhisper v2 from official model id")
        return 2

    from whisper_timestamped.crisperwhisper.hub_offline import (
        call_with_local_files_fallback,
    )

    for filename in ("added_tokens.json", "vocab.json", "tokenizer.json"):
        try:
            from huggingface_hub import hf_hub_download

            local = Path(
                call_with_local_files_fallback(
                    hf_hub_download, model_name_or_path, filename,
                )
            )
            result = _check_local_dir(local.parent)
            if result is not None:
                logger.info("Detected CrisperWhisper v%d from HF Hub", result)
                return result
        except Exception:
            continue

    logger.warning(
        "Could not detect model version for %s; assuming v2.",
        model_name_or_path,
    )
    return 2
