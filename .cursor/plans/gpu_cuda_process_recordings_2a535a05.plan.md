---
name: GPU/CUDA process_recordings
overview: Enable GPU/CUDA for process_recordings by (1) installing the CUDA build of PyTorch on Windows via uv index configuration, and (2) making device selection explicit and visible in the script with an optional override and clearer logging.
todos: []
isProject: false
---

# GPU/CUDA mode for process_recordings

## Current state

- [scripts/process_recordings.py](C:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\whisper-timestamped\scripts\process_recordings.py) already uses CUDA when available (lines 214–217):
  - `device = "cuda" if torch.cuda.is_available() else "cpu"`
  - `whisper.load_model(..., device=device)`
- On **Windows**, PyPI serves **CPU-only** PyTorch wheels, so `uv sync` typically installs a build where `torch.cuda.is_available()` is `False`. That is why processing falls back to CPU even with a GPU.

## 1. Install PyTorch with CUDA via uv (pyproject.toml)

Use uv’s PyTorch index so that on Windows (and Linux) the project gets the CUDA build instead of the default CPU build.

- Add a dedicated PyTorch index (e.g. CUDA 12.4 or 12.1) in `[[tool.uv.index]]` with `explicit = true`.
- In `[tool.uv.sources]`, point `torch`, `torchaudio`, and `torchvision` to that index only when `sys_platform == 'win32'` or `sys_platform == 'linux'` (macOS keeps using PyPI; no CUDA wheels there).
- Pick one CUDA version to support (e.g. **cu124** for current drivers; **cu121** if you need older driver compatibility). Index URLs: [uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/#installing-pytorch).

Example addition (concept; exact version constraints stay as in your current `dependencies`):

```toml
[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true

[tool.uv.sources]
# ... existing openai-whisper, phopylslhelper ...
torch = [{ index = "pytorch-cu124", marker = "sys_platform == 'win32' or sys_platform == 'linux'" }]
torchaudio = [{ index = "pytorch-cu124", marker = "sys_platform == 'win32' or sys_platform == 'linux'" }]
torchvision = [{ index = "pytorch-cu124", marker = "sys_platform == 'win32' or sys_platform == 'linux'" }]
```

After this, run `uv lock` and `uv sync --all-extras` (or your usual sync). On Windows/Linux, torch will be resolved from the CUDA index; `torch.cuda.is_available()` should then be `True` when a supported GPU/driver is present.

**Optional:** If you prefer not to change the default install, you can instead add an optional extra (e.g. `cuda`) that uses the CUDA index and document “use `uv sync --extra cuda` for GPU”. That requires `[tool.uv.sources]` with an `extra = "cuda"` marker and a `[tool.uv.conflicts]` between `cpu` and `cuda` if you also add a CPU extra.

## 2. Script: explicit device and logging (process_recordings.py)

- Add an optional parameter to `process_recordings()`: e.g. `device: Optional[str] = None`. Semantics:
  - `None`: keep current behavior (use CUDA if available, else CPU).
  - `"cuda"`: use GPU (fail or warn if not available).
  - `"cpu"`: force CPU.
- When the chosen device is CUDA (either auto-selected or forced), log the GPU name once after model load, e.g. `print(f"Using GPU: {torch.cuda.get_device_name(0)}")`, so it’s clear that GPU is in use.
- If you later add a small CLI (e.g. `argparse` in `if __name__ == "__main__"`) for `recordings_dir` / `output_dir`, add a `--device` option (e.g. `cuda` | `cpu` | `auto`) and pass it into `process_recordings(..., device=...)`.

No change to the rest of the pipeline (e.g. VAD preload, `whisper.transcribe`) is required; the existing `device` passed to `load_model` is already used by the library for inference.

## 3. Verify

- On a Windows machine with an NVIDIA GPU and up-to-date driver, run `uv sync` (with the new index/sources), then:
  - `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"`
- Run `process_recordings` and confirm the log shows `device=cuda` and “Using GPU: …”. Processing should be noticeably faster for long recordings.

## Summary


| Area                      | Action                                                                                                                                                         |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **pyproject.toml**        | Add `[[tool.uv.index]]` for PyTorch CUDA (e.g. cu124) and `[tool.uv.sources]` for `torch`, `torchaudio`, `torchvision` with platform marker for Windows/Linux. |
| **process_recordings.py** | Add optional `device` parameter; when device is CUDA, log GPU name; optionally add `--device` to a future CLI.                                                 |


Result: “Try processing using GPU/CUDA” is achieved by (1) installing the CUDA build of PyTorch on Windows via uv, and (2) making device selection and GPU usage visible in the script.