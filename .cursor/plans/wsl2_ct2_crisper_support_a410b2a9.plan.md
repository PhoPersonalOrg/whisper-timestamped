---
name: WSL2 CT2 Crisper support
overview: Restore CrisperWhisper’s CTranslate2 modules and enable them on Linux/WSL2 via a Linux-only extra, while Windows keeps the transformers fallback and its current live (faster-whisper) install.
todos: []
isProject: false
---

# Add CT2 CrisperWhisper for WSL2 (keep Windows transformers)

## Constraint

`ctranslate2-crisperwhisper` is Linux-only. Upstream `ctranslate2` (from `faster-whisper`) occupies the same import namespace and clobbers the fork. Windows must keep transformers + working live transcription.

## Dependency layout

In [`pyproject.toml`](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\whisper-timestamped\pyproject.toml):

- Change core `faster-whisper` to Windows-only so a default Linux/WSL sync does not pull upstream `ctranslate2`:

```toml
"faster-whisper>=1.2.0; sys_platform == 'win32'",
```

- Keep `faster-whisper` in the existing `live` extra (Linux live users: `uv sync --extra live`).
- Add Linux-only extra:

```toml
crisper_ct2 = [
  "ctranslate2-crisperwhisper>=4.7.1.post3,<5; sys_platform == 'linux'",
  "nvidia-cublas-cu12; sys_platform == 'linux'",
]
```

Document: do not install `live` and `crisper_ct2` in the same env (fork vs upstream clash). WSL2 batch: `uv sync --extra crisper_ct2`.

Windows `uv sync` stays transformers + faster-whisper; unchanged for live.

## Vendor CT2 modules

Copy from CrisperWhisper into [`whisper_timestamped/crisperwhisper/`](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\whisper-timestamped\whisper_timestamped\crisperwhisper) and rewrite imports to `whisper_timestamped.crisperwhisper.*`:

- `engine.py`, `converter.py`, `speculative.py`, `hallucination.py`, `features.py`

Update [`model.py`](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\whisper-timestamped\whisper_timestamped\crisperwhisper\model.py) `_resolve_backend("auto")` to prefer CT2 only when the **fork APIs** exist (not merely `import ctranslate2`), else transformers — so a botched Linux install with upstream CT2 falls back instead of crashing mid-init.

## Adapter: auto runtime

In [`crisper_backend.py`](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\whisper-timestamped\whisper_timestamped\crisper_backend.py):

- Add `crisper_runtime: str = "auto"` to `load_crisper_model` (`"auto"` | `"ct2"` | `"transformers"`).
- Resolve: `ct2` if fork APIs + vendored `engine` available; else `transformers`. Explicit `"ct2"` raises a clear error on Windows / missing fork.
- When runtime is `ct2`: load with `backend="ct2"`, `draft_model="turbo"` (for `medium`/`large`/`small` shorthands; skip draft if already loading turbo).
- Store resolved runtime on `CrisperWhisperAsTimestamped`.
- In `transcribe_crisper`: if runtime is `ct2`, pass `speculative_decoding=True`; transformers path unchanged.

Wire `crisper_runtime` through `load_model` / `transcribe_timestamped` / CLI (`--crisper_runtime`).

## process_recordings

Pass `crisper_runtime="auto"` (default) so WSL2+extra uses CT2 and Windows stays transformers. Log which runtime was selected after load.

## Verify

- Windows: `uv sync` still works; `backend=crisperwhisper` loads transformers.
- WSL2: `uv sync --extra crisper_ct2`, then load with auto → CT2 + speculative; first run converts weights into `~/.cache/crisperwhisper/`.
- Unit test: mock/fork-availability helper + mapper still pass without downloading models.