---
name: CrisperWhisper optional mode
overview: Vendor CrisperWhisper’s PyTorch inference path into whisper-timestamped and expose it as an opt-in backend. The existing openai-whisper transcribe API, return shape, and live faster-whisper path stay the default so current callers keep working. Then make `uv sync` succeed on Windows.
todos:
  - id: vendor-cw
    content: Copy transformers-path CrisperWhisper modules into whisper_timestamped/crisperwhisper and rewrite imports
    status: completed
  - id: adapter
    content: Add crisper_backend adapter and opt-in load_model/transcribe/CLI wiring that preserves the existing result dict
    status: completed
  - id: deps-sync
    content: Add soxr and accelerate, run uv sync on Windows, and verify the package imports
    status: completed
  - id: tests-readme
    content: Add a no-download mapper unit test and a short README note on the optional mode and weight license
    status: completed
isProject: false
---

# Optional CrisperWhisper mode in whisper-timestamped

## Constraint that decides the design

whisper-timestamped already depends on `faster-whisper`, which installs upstream `ctranslate2`. CrisperWhisper’s fast backend needs the separate `ctranslate2-crisperwhisper` fork, which occupies the same `ctranslate2` import path and is published as Linux wheels only. Installing it would break live transcription and would not install on Windows.

The accuracy improvements (verbatim/intended prompts, supervised cross-attention word timing, longform continuation, loop repair) all run on CrisperWhisper’s **transformers** backend, which this repo already has `torch` and `transformers` for. Speculative decoding stays out of scope.

Default behavior does not change: `load_model` still uses `openai-whisper`, and [`whisper_timestamped/live.py`](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\whisper-timestamped\whisper_timestamped\live.py) stays on faster-whisper.

```mermaid
flowchart TD
  call[transcribe or CLI]
  call --> which{backend}
  which -->|openai-whisper default| existing[existing DTW timestamp path]
  which -->|transformers| existing
  which -->|crisperwhisper| cw[vendored CrisperWhisperModel backend=transformers]
  cw --> map[map text chunks words into segments]
  existing --> out["dict text language segments words"]
  map --> out
```

## Vendor the inference code

Copy the MIT-licensed CrisperWhisper package into [`whisper_timestamped/crisperwhisper/`](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\whisper-timestamped\whisper_timestamped\crisperwhisper), rewrite imports from `crisperwhisper` to `whisper_timestamped.crisperwhisper`, and keep the Nyra MIT notice in that folder.

Copy the modules the transformers path actually imports: `model.py`, `transformers_engine.py`, `interfaces.py`, `prompt.py`, `audio.py`, `word_timing.py`, `loop_detection.py`, `fallback.py`, `result.py`, `version.py`, `forced_align.py`, `_nvidia_libs.py`, and `longform/`. Leave out CT2-only modules (`engine.py`, `converter.py`, `speculative.py`, `hallucination.py`, `features.py`) so nothing imports `ctranslate2` from this tree. The package `__init__` should export `CrisperWhisperModel` without probing the CT2 fork.

## Opt-in API, same result dict

Add a small adapter, [`whisper_timestamped/crisper_backend.py`](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\whisper-timestamped\whisper_timestamped\crisper_backend.py):

- `load_crisper_model(name, device=...)` builds `CrisperWhisperModel(..., backend="transformers")`. Shorthands `large`, `turbo`, `medium`, `small` (and `*_pro`) resolve to `nyralabs/CrisperWhisper2.0_*`. On CPU, use float32; on CUDA, float16.
- `transcribe_crisper(...)` calls `transcribe` with `word_timestamps=True`, `hallucination_mitigation=True`, and `longform_strategy="continuation"`. Optional `crisper_mode` is `"verbatim"` (default) or `"intended"`.
- Map the result onto the existing schema from [`tests/json_schema.json`](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\whisper-timestamped\tests\json_schema.json): top-level `text` and `language`; `segments[]` from CrisperWhisper `chunks` (one segment for audio under 30s); each word as `{text, start, end}` under `segments[].words`. CrisperWhisper does not emit token ids, logprobs, or `language_probs`, so those keys stay absent. Word key is `text`, matching the current API (not CrisperWhisper’s `word`).

Wire it in [`whisper_timestamped/transcribe.py`](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\whisper-timestamped\whisper_timestamped\transcribe.py) **before** the `model.dims` / `model.device` reads around lines 249–257, which a CrisperWhisper model does not have:

- `load_model(..., backend="crisperwhisper")` returns the wrapper. Unknown backend values still raise, as they do today.
- `transcribe_timestamped` gains optional `backend=None` and `crisper_mode="verbatim"`. A string model name still loads openai-whisper unless `backend="crisperwhisper"`. If the loaded object is the Crisper wrapper, return the adapter result and skip DTW. If `vad` is set, keep the existing `remove_non_speech` / timestamp remap so that flag still works.
- CLI `--backend` choices become `openai-whisper`, `transformers`, `crisperwhisper`. Add `--crisper_mode` with choices `verbatim` and `intended`. CLI default `--model small` already matches a CrisperWhisper shorthand. Names that are only stock Whisper sizes (`tiny`, `base`, `large-v2`, …) get a clear error listing the Crisper shorthands.

[`whisper_timestamped/__init__.py`](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\whisper-timestamped\whisper_timestamped\__init__.py) keeps exporting `transcribe` and `load_model` unchanged.

## Dependencies and Windows `uv sync`

In [`pyproject.toml`](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\whisper-timestamped\pyproject.toml), add the two runtime packages the vendored code needs that are not already locked: `soxr>=0.3` (16 kHz resample) and `accelerate>=0.26` (`from_pretrained(..., low_cpu_mem_usage=True)`). Do not add `ctranslate2-crisperwhisper`. Keep `requires-python` at `>=3.10,<3.11`.

Run `uv sync` in the whisper-timestamped repo on Windows, then confirm `uv run python -c "import whisper_timestamped"` succeeds. That is the build check. Loading `nyralabs/CrisperWhisper2.0_*` weights is a separate download and is not required for sync.

Add a unit test that feeds a fake `TranscriptionResult` (short clip and a two-chunk longform result) through the mapper and checks `text`, `language`, `segments[].words[].text`, and monotonic times, with no model download.

Note in the README that this mode is opt-in, uses CrisperWhisper 2.0 weights under Nyra’s non-commercial research license, and that the vendored inference code is MIT.
