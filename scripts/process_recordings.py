import os
import re
import signal
import sys
# import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from whisper.utils import str2bool, optional_float, optional_int
import whisper_timestamped as whisper
from whisper_timestamped.transcribe import write_csv, flatten, remove_keys, get_vad_segments
from whisper_timestamped.parse_video_filename import build_EDF_compatible_video_filename, parse_video_filename
from whisper_timestamped.recording_formats import get_format
# from whisper_timestamped import remove_non_speech
from whisper_timestamped.transcribe import remove_non_speech


def _running_in_wsl() -> bool:
    """True when running inside WSL (not native Windows or bare Linux)."""
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        with open("/proc/version", encoding="utf-8") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def host_path(path: Union[str, Path]) -> Path:
    """Map Windows drive paths ↔ WSL ``/mnt/<drive>/...`` for the current host.

    Convert before ``.resolve()``. Unrelated paths are returned unchanged.
    """
    s = str(path).replace("\\", "/")
    drive_m = re.match(r"^([A-Za-z]):/(.*)$", s)
    if drive_m:
        letter, rest = drive_m.group(1), drive_m.group(2)
        if _running_in_wsl():
            return Path(f"/mnt/{letter.lower()}/{rest}")
        return Path(f"{letter}:/{rest}")

    mnt_m = re.match(r"^/mnt/([a-zA-Z])/(.*)$", s)
    if mnt_m and sys.platform == "win32" and not _running_in_wsl():
        letter, rest = mnt_m.group(1).upper(), mnt_m.group(2)
        return Path(f"{letter}:/{rest}")

    return Path(path)

try:
    # Old whisper version # Before https://github.com/openai/whisper/commit/da600abd2b296a5450770b872c3765d0a5a5c769
    from whisper.utils import write_txt, write_srt, write_vtt
    write_tsv = lambda transcript, file: write_csv(transcript, file, sep="\t", header=True, text_first=False, format_timestamps=lambda x: round(1000 * x))

except ImportError:
    # New whisper version
    from whisper.utils import get_writer

    def do_write(transcript, file, output_format):
        writer = get_writer(output_format, os.path.curdir)
        try:
            return writer.write_result({"segments": list(transcript)}, file, {
                "highlight_words": False,
                "max_line_width": None,
                "max_line_count": None,
            })
        except TypeError:
            # Version <= 20230314
            return writer.write_result({"segments": transcript}, file)
    def get_do_write(output_format):
        return lambda transcript, file: do_write(transcript, file, output_format)

    write_txt = get_do_write("txt")
    write_srt = get_do_write("srt")
    write_vtt = get_do_write("vtt")
    write_tsv = get_do_write("tsv")
    


# Segment-level suffix per output format; whisper-timestamped uses .words.json for JSON.
_OUTPUT_SUFFIX_BY_FORMAT = {
    "json": ".words.json",
    "csv": ".csv",
    "txt": ".txt",
    "vtt": ".vtt",
    "srt": ".srt",
    "tsv": ".tsv",
}

# All write_results keys → on-disk suffix (for filling transcript_* on skip/re-run).
_TRANSCRIPT_OUTPUT_SPECS: List[Tuple[str, str]] = [
    ("json", ".words.json"),
    ("csv", ".csv"),
    ("words.csv", ".words.csv"),
    ("txt", ".txt"),
    ("vtt", ".vtt"),
    ("words.vtt", ".words.vtt"),
    ("srt", ".srt"),
    ("words.srt", ".words.srt"),
    ("tsv", ".tsv"),
    ("words.tsv", ".words.tsv"),
]

# Ctrl+C: 1× soft-stop after current file; 3× within this window force-aborts.
_SIGINT_WINDOW_S = 2.0


class _GracefulInterruptState:
    """SIGINT: soft-stop after current recording; 3 quick presses force-abort."""

    def __init__(self) -> None:
        self.stop_after_current = False
        self.force_abort = False
        self._press_times: List[float] = []
        self.current_base_name: Optional[str] = None
        self.output_dir: Optional[Path] = None
        self._prev_handler = None

    def install(self) -> None:
        self._prev_handler = signal.signal(signal.SIGINT, self._on_sigint)

    def restore(self) -> None:
        if self._prev_handler is not None:
            signal.signal(signal.SIGINT, self._prev_handler)
            self._prev_handler = None

    def _on_sigint(self, signum, frame) -> None:
        now = time.monotonic()
        self._press_times = [t for t in self._press_times if now - t < _SIGINT_WINDOW_S]
        self._press_times.append(now)
        n = len(self._press_times)
        if n >= 3:
            self.force_abort = True
            self.stop_after_current = True
            print(
                "\n  ! Force-abort: discarding in-flight work and stopping...",
                file=sys.stderr,
                flush=True,
            )
            # Restore default so a stuck abort can still be killed.
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            raise KeyboardInterrupt
        self.stop_after_current = True
        if n == 1:
            print(
                "\n  ! Ctrl+C: finishing current recording, then stopping. "
                "Press Ctrl+C twice more quickly to force-abort.",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                "\n  ! Ctrl+C again (quickly) to force-abort current recording.",
                file=sys.stderr,
                flush=True,
            )


def _discard_transcript_outputs(output_dir: Path, base_name: str) -> List[Path]:
    """Remove any transcript outputs for base_name so the file can be reprocessed."""
    removed: List[Path] = []
    output_file_path = output_dir.joinpath(base_name)
    seen = set()
    for _, suffix in _TRANSCRIPT_OUTPUT_SPECS:
        a_file = output_file_path.with_suffix(suffix)
        key = a_file.as_posix()
        if key in seen:
            continue
        seen.add(key)
        if a_file.exists() and a_file.is_file():
            try:
                a_file.unlink()
                removed.append(a_file)
            except OSError as e:
                print(f"  ~ Could not remove partial output {a_file.name}: {e}")
        ## END if a_file exists....
    ## END for _, suffix in _TRANSCRIPT_OUTPUT_SPECS....

    return removed


def _transcript_column_for_key(key: str) -> str:
    return f"transcript_{key.replace('.', '_')}"


def collect_extant_transcript_paths(output_dir: Path, base_name: str) -> Dict[str, str]:
    """Map transcript_* column -> absolute path for outputs that already exist."""
    output_file_path = output_dir.joinpath(base_name)
    cols: Dict[str, str] = {}
    for key, suffix in _TRANSCRIPT_OUTPUT_SPECS:
        a_file = output_file_path.with_suffix(suffix)
        col = _transcript_column_for_key(key)
        if a_file.exists() and a_file.is_file():
            cols[col] = str(a_file.resolve())
        else:
            cols[col] = ""
        ## END if a_file exists....
    ## END for key, suffix in _TRANSCRIPT_OUTPUT_SPECS....

    return cols


def flatten_write_results_to_transcript_cols( curr_output_files_dict: dict, base_name: str) -> Dict[str, str]:
    """Map write_results nested dict to transcript_* absolute path strings."""
    cols: Dict[str, str] = {
        _transcript_column_for_key(key): "" for key, _ in _TRANSCRIPT_OUTPUT_SPECS
    }
    for key, by_base in curr_output_files_dict.items():
        path = by_base.get(base_name)
        col = _transcript_column_for_key(key)
        if path is not None:
            cols[col] = str(Path(path).resolve())
        ## END if path is not None....
    ## END for key, by_base in curr_output_files_dict.items()....

    return cols


def find_extant_output_files(output_dir: Path, base_name: str, output_formats = ['json', 'csv', 'srt', 'vtt', 'txt']) -> List[Path]:
    """ found any of the output files that would be created upon transcode completion in the output_dir 

    found_output_files: List[Path] = find_extant_output_files(output_dir=output_dir, base_name=base_name, output_formats=output_formats)

    """
    output_file_path: Path = output_dir.joinpath(base_name)
    found_output_files: List[Path] = []
    for fmt in output_formats:
        suffix = _OUTPUT_SUFFIX_BY_FORMAT.get(fmt, f".{fmt}")
        a_file: Path = output_file_path.with_suffix(suffix)
        if a_file.exists() and a_file.is_file():
            found_output_files.append(a_file)

    return found_output_files


def _register_output(output_files: dict, key: str, base_name: str, path: Path) -> None:
    output_files.setdefault(key, {})[base_name] = path


def write_results(result, output_dir: Path, base_name: str, output_formats = ['json', 'csv', 'srt', 'vtt', 'txt']):
    """ Writes the results object out to disk
    base_name = video_file.stem
    output_files = write_results(result, output_dir=output_dir, base_name=base_name)

    """
    output_file_path: Path = output_dir.joinpath(base_name)
    print(F'building output files for output_file_path: "{output_file_path.as_posix()}"')
    output_files: dict = {}

    if "json" in output_formats:
        try:
            a_file = output_file_path.with_suffix(".words.json")
            with open(a_file, "w", encoding="utf-8") as js:
                json.dump(result, js, indent=2, ensure_ascii=False)
            _register_output(output_files, "json", base_name, a_file)
            print(f"  ✓ Saved: {a_file.name}")
        except Exception as e:
            print(f"  ✗ Error saving JSON: {e}")

    if "csv" in output_formats:
        try:
            a_file = output_file_path.with_suffix(".csv")
            with open(a_file, "w", encoding="utf-8") as csv:
                write_csv(result["segments"], file=csv, header=True)
            _register_output(output_files, "csv", base_name, a_file)
            print(f"  ✓ Saved: {a_file.name}")
        except Exception as e:
            print(f"  ✗ Error saving CSV: {e}")

        try:
            a_file = output_file_path.with_suffix(".words.csv")
            with open(a_file, "w", encoding="utf-8") as csv:
                write_csv(flatten(result["segments"], "words"), file=csv, header=True)
            _register_output(output_files, "words.csv", base_name, a_file)
            print(f"  ✓ Saved: {a_file.name}")
        except Exception as e:
            print(f"  ✗ Error saving words CSV: {e}")

    if "txt" in output_formats:
        try:
            a_file = output_file_path.with_suffix(".txt")
            with open(a_file, "w", encoding="utf-8") as txt:
                write_txt(result["segments"], file=txt)
            _register_output(output_files, "txt", base_name, a_file)
            print(f"  ✓ Saved: {a_file.name}")
        except Exception as e:
            print(f"  ✗ Error saving TXT: {e}")

    if "vtt" in output_formats:
        try:
            a_file = output_file_path.with_suffix(".vtt")
            with open(a_file, "w", encoding="utf-8") as vtt:
                write_vtt(remove_keys(result["segments"], "words"), file=vtt)
            _register_output(output_files, "vtt", base_name, a_file)
            print(f"  ✓ Saved: {a_file.name}")
        except Exception as e:
            print(f"  ✗ Error saving VTT: {e}")

        try:
            a_file = output_file_path.with_suffix(".words.vtt")
            with open(a_file, "w", encoding="utf-8") as vtt:
                write_vtt(flatten(result["segments"], "words"), file=vtt)
            _register_output(output_files, "words.vtt", base_name, a_file)
            print(f"  ✓ Saved: {a_file.name}")
        except Exception as e:
            print(f"  ✗ Error saving words VTT: {e}")

    if "srt" in output_formats:
        try:
            a_file = output_file_path.with_suffix(".srt")
            with open(a_file, "w", encoding="utf-8") as srt:
                write_srt(remove_keys(result["segments"], "words"), file=srt)
            _register_output(output_files, "srt", base_name, a_file)
            print(f"  ✓ Saved: {a_file.name}")
        except Exception as e:
            print(f"  ✗ Error saving SRT: {e}")

        try:
            a_file = output_file_path.with_suffix(".words.srt")
            with open(a_file, "w", encoding="utf-8") as srt:
                write_srt(flatten(result["segments"], "words"), file=srt)
            _register_output(output_files, "words.srt", base_name, a_file)
            print(f"  ✓ Saved: {a_file.name}")
        except Exception as e:
            print(f"  ✗ Error saving words SRT: {e}")

    if "tsv" in output_formats:
        try:
            a_file = output_file_path.with_suffix(".tsv")
            with open(a_file, "w", encoding="utf-8") as csv:
                write_tsv(result["segments"], file=csv)
            _register_output(output_files, "tsv", base_name, a_file)
            print(f"  ✓ Saved: {a_file.name}")
        except Exception as e:
            print(f"  ✗ Error saving TSV: {e}")

        try:
            a_file = output_file_path.with_suffix(".words.tsv")
            with open(a_file, "w", encoding="utf-8") as csv:
                write_tsv(flatten(result["segments"], "words"), file=csv)
            _register_output(output_files, "words.tsv", base_name, a_file)
            print(f"  ✓ Saved: {a_file.name}")
        except Exception as e:
            print(f"  ✗ Error saving words TSV: {e}")

    return output_files


def process_recordings(
    recordings_dir: Optional[Path] = None,
    output_dir=None,
    video_extensions=['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v'],
    model_path_root: Path = Path(r'F:\AITEMP\whisper_models'),
    backend: str = "openai-whisper",
    model_name: str = None,
    crisper_mode: str = "verbatim",
    crisper_runtime: str = "auto",
    filelist_csv: Optional[Path] = None,
):
    has_filelist = filelist_csv is not None
    has_dir = recordings_dir is not None
    if has_filelist == has_dir:
        raise ValueError(
            "Provide exactly one of filelist_csv or recordings_dir "
            f"(got filelist_csv={filelist_csv!r}, recordings_dir={recordings_dir!r})"
        )
    ## END if has_filelist == has_dir....

    filelist_df: Optional[pd.DataFrame] = None
    filelist_path: Optional[Path] = None
    # jobs: (audio_path, base_name, optional filelist row index)
    jobs: List[Tuple[Path, str, Optional[object]]] = []

    if has_filelist:
        filelist_path = host_path(filelist_csv).resolve()
        if not filelist_path.is_file():
            raise FileNotFoundError(f"filelist_csv not found: {filelist_path}")
        filelist_df = pd.read_csv(filelist_path)
        if "full_path" not in filelist_df.columns:
            raise ValueError(
                f"filelist_csv missing required 'full_path' column: {filelist_path}"
            )
        print(f'processing_recordings for filelist_csv: "{filelist_path.as_posix()}"...')

        if output_dir is None:
            # filelists/foo.csv → sibling transcriptions/; else beside the CSV
            if filelist_path.parent.name.lower() == "filelists":
                output_dir = filelist_path.parent.parent / "transcriptions"
            else:
                output_dir = filelist_path.parent / "transcriptions"
            ## END if under filelists/....

            output_dir = host_path(output_dir).resolve()
        else:
            output_dir = host_path(output_dir).resolve()
        ## END if output_dir is None....

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f'\t transcriptions will output to output_dir: "{output_dir.as_posix()}"')

        alias_dir = output_dir.parent / "edf_video_aliases"

        for idx, row in filelist_df.iterrows():
            # Skip non-keepers flagged by extract_m4a_creation_times (bool or "true"/"1"/"yes")
            if "is_duplicate" in filelist_df.columns:
                dup_raw = row.get("is_duplicate")
                if pd.notna(dup_raw) and (
                    dup_raw is True
                    or str(dup_raw).strip().lower() in ("true", "1", "yes")
                ):
                    print(f"  ~ Skipping duplicate row index={idx}")
                    continue
                ## END if truthy is_duplicate....
            ## END if is_duplicate column....

            full_path_raw = row.get("full_path")
            if pd.isna(full_path_raw) or not str(full_path_raw).strip():
                print(f"  ! Missing full_path for row index={idx}")
                continue
            ## END if full_path missing....

            audio_path = host_path(str(full_path_raw).strip()).resolve()
            if not audio_path.is_file():
                print(f"  ! Missing audio file: {audio_path}")
                continue
            ## END if audio missing....

            name_raw = row.get("name") if "name" in filelist_df.columns else None
            if name_raw is not None and pd.notna(name_raw) and str(name_raw).strip():
                base_name = Path(str(name_raw).strip()).stem
            else:
                base_name = audio_path.stem
            ## END if name present....

            jobs.append((audio_path, base_name, idx))
        ## END for idx, row in filelist_df.iterrows()....

        # Ensure transcript_* columns exist up front.
        for key, _ in _TRANSCRIPT_OUTPUT_SPECS:
            col = _transcript_column_for_key(key)
            if col not in filelist_df.columns:
                filelist_df[col] = ""
            ## END if col missing....
        ## END for key, _ in _TRANSCRIPT_OUTPUT_SPECS....
    else:
        recordings_dir = host_path(recordings_dir).resolve()
        print(f'processing_recordings for recordings_dir: "{recordings_dir.as_posix()}"...')
        if output_dir is None:
            output_dir = recordings_dir.joinpath('transcriptions').resolve()
        else:
            output_dir = host_path(output_dir).resolve()
        ## END if output_dir is None....

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f'\t transcriptions will output to output_dir: "{output_dir.as_posix()}"')

        video_files: List[Path] = []
        for ext in video_extensions:
            video_files.extend(recordings_dir.glob(f"*{ext}"))
            video_files.extend(recordings_dir.glob(f"*{ext.upper()}"))
        ## END for ext in video_extensions....

        alias_dir = recordings_dir.parent / "edf_video_aliases"

        for video_file in video_files:
            jobs.append((video_file, video_file.stem, None))
        ## END for video_file in video_files....
    ## END if has_filelist....

    if not jobs:
        print("No audio/video files to process")
        return {}

    print(f"Found {len(jobs)} file(s) to process")

    # Load the model once (after file discovery so progress is visible sooner)
    if model_name is None:
        model_name = "medium" if backend == "crisperwhisper" else "medium.en"
    model_path_root = host_path(model_path_root).resolve()
    # CrisperWhisper weights come from HuggingFace cache, not model_path_root.
    if backend != "crisperwhisper":
        assert model_path_root.exists()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"Loading Whisper model {model_name!r} (backend={backend}, device={device}"
        + (f", model_path_root='{model_path_root.as_posix()}'" if backend != "crisperwhisper" else "")
        + ")..."
    )
    t0_model = time.perf_counter()
    model = whisper.load_model(
        model_name,
        download_root=str(model_path_root),
        device=device,
        backend=backend,
        crisper_runtime=crisper_runtime,
    )
    runtime_note = ""
    if backend == "crisperwhisper" and hasattr(model, "runtime"):
        runtime_note = f" crisper_runtime={model.runtime!r}"
    print(
        f"Whisper model loaded.{runtime_note} "
        f"(Model load: {time.perf_counter() - t0_model:.1f}s)"
    )

    # Preload Silero VAD so first-file transcribe does not stall with no progress
    print("Loading Silero VAD...")
    get_vad_segments(torch.zeros(16000, dtype=torch.float32), method="silero")  # 1s at 16kHz (Whisper SAMPLE_RATE)
    print("Done.")

    output_files: dict = {}
    first_file_timed = True
    failed_files: List[Path] = []
    interrupt = _GracefulInterruptState()
    interrupt.output_dir = output_dir
    interrupt.install()

    def _persist_filelist_row(row_index: object, transcript_cols: Dict[str, str]) -> None:
        if filelist_df is None or filelist_path is None:
            return
        for col, path_str in transcript_cols.items():
            filelist_df.at[row_index, col] = path_str
        ## END for col, path_str in transcript_cols.items()....

        filelist_df.to_csv(filelist_path, index=False, encoding="utf-8")

    try:
        for audio_path, base_name, row_index in jobs:
            if interrupt.stop_after_current:
                print("\nSoft-stop: not starting further recordings after Ctrl+C.")
                break
            ## END if interrupt.stop_after_current....

            print(f"\nProcessing: {audio_path.name} (base_name={base_name!r})")
            interrupt.current_base_name = base_name
            try:
                ## try making a symlink with an EDF+ compatible formatted name: https://www.edfplus.info/specs/video.html
                try:
                    edf_compatible_name = build_EDF_compatible_video_filename(audio_path.name)
                    print(f'\tedf_compatible_name: "{edf_compatible_name}"')
                    alias_dir.mkdir(exist_ok=True)
                    edf_compatible_path = alias_dir / edf_compatible_name
                    if not edf_compatible_path.exists():
                        edf_compatible_path.symlink_to(audio_path.resolve())
                except (ValueError, OSError) as e:
                    print(f"  ~ Skipping EDF alias for {audio_path.name}: {e}")

                found_output_files: List[Path] = find_extant_output_files(
                    output_dir=output_dir, base_name=base_name
                )
                if found_output_files:
                    print(
                        f"  ✗ Skipping {audio_path.name} as its outputs already exist: "
                        f"{found_output_files}"
                    )
                    if row_index is not None:
                        _persist_filelist_row(
                            row_index,
                            collect_extant_transcript_paths(output_dir, base_name),
                        )
                    ## END if row_index is not None....

                    interrupt.current_base_name = None
                    continue
                ## END if found_output_files....

                if first_file_timed:
                    print("  Running VAD and transcription...")
                print("  Loading audio...")
                t0_audio = time.perf_counter()
                audio = whisper.load_audio(str(audio_path))
                print("  Audio loaded.")
                if first_file_timed:
                    print(f"  First file load_audio: {time.perf_counter() - t0_audio:.1f}s")

                t0_transcribe = time.perf_counter()
                result = whisper.transcribe(
                    model,
                    audio,
                    language="en",
                    vad="silero",
                    remove_empty_words=True,
                    crisper_mode=crisper_mode,
                )
                if first_file_timed:
                    print(f"  First file transcribe: {time.perf_counter() - t0_transcribe:.1f}s")
                    first_file_timed = False

                curr_output_files_dict = write_results(
                    result, output_dir=output_dir, base_name=base_name
                )
                for k, curr_out_files_dict in curr_output_files_dict.items():
                    if k not in output_files:
                        output_files[k] = dict()
                    output_files[k].update(**curr_out_files_dict)
                ## END for k, curr_out_files_dict in curr_output_files_dict.items()....

                if row_index is not None:
                    _persist_filelist_row(
                        row_index,
                        flatten_write_results_to_transcript_cols(
                            curr_output_files_dict, base_name
                        ),
                    )
                ## END if row_index is not None....

                interrupt.current_base_name = None
                if interrupt.stop_after_current:
                    print("\nSoft-stop: finished current recording after Ctrl+C; stopping.")
                    break
                ## END if interrupt.stop_after_current....

            except KeyboardInterrupt:
                # Force-abort (3× Ctrl+C) or interrupt that escaped soft-stop handling.
                if interrupt.current_base_name is not None:
                    removed = _discard_transcript_outputs(
                        output_dir, interrupt.current_base_name
                    )
                    if removed:
                        print(
                            f"  Discarded partial outputs for {interrupt.current_base_name!r}: "
                            f"{[p.name for p in removed]}"
                        )
                    else:
                        print(
                            f"  No transcript outputs to discard for "
                            f"{interrupt.current_base_name!r} (safe to reprocess)."
                        )
                    ## END if removed....

                    interrupt.current_base_name = None
                ## END if interrupt.current_base_name....

                raise
            except Exception as e:
                failed_files.append(audio_path)
                print(f"  ✗ Error processing {audio_path.name}: [{type(e).__name__}] {e}")
                if row_index is not None:
                    # Leave transcript_* empty / unchanged for this failure; still flush CSV.
                    filelist_df.to_csv(filelist_path, index=False, encoding="utf-8")
                ## END if row_index is not None....

                interrupt.current_base_name = None
                continue
            ## END try/except per file....
        ## END for audio_path, base_name, row_index in jobs....
    finally:
        interrupt.restore()
    ## END try/finally interrupt handler....

    if failed_files:
        print(f"\nProcessing complete with {len(failed_files)} failed file(s): {[f.name for f in failed_files]}")
    if filelist_path is not None:
        print(f"Filelist updated: {filelist_path}")
    if interrupt.stop_after_current and not interrupt.force_abort:
        print(f"\nStopped after Ctrl+C. Output files saved to: {output_dir.resolve()}")
    else:
        print(f"\nProcessing complete! Output files saved to: {output_dir.resolve()}")
    ## END if soft-stop vs complete....

    return output_files


if __name__ == "__main__":
    # Switch formats here instead of commenting path blocks.
    # Known ids: debut | rec_continuous | ios_whisper_app | just_press_record | voice_memos
    ACTIVE_FORMAT = "just_press_record"
    fmt = get_format(ACTIVE_FORMAT)
    process_recordings_kwargs = fmt.process_recordings_kwargs()
    print(
        f"Active recordings format: {fmt.id} ({fmt.label}) -> "
        f"{process_recordings_kwargs}"
    )

    try:
        output_files = process_recordings(
            **process_recordings_kwargs,
            backend="crisperwhisper",
            model_name="medium",
            crisper_mode="verbatim",
            crisper_runtime="auto",  # CT2 on WSL2 with --extra crisper_ct2; transformers on Windows
        )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    print(f'All processing complete! output_files: {output_files}\n\ndone.')

