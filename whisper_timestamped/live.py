import argparse
import json
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import numpy as np

try:
    import sounddevice as sd
    import soundfile as sf
except Exception as e:  # pragma: no cover - optional at import time
    sd = None
    sf = None

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None

try:
    from pylsl import StreamInfo, StreamOutlet
    LSL_AVAILABLE = True
except Exception:
    LSL_AVAILABLE = False


def _startup_mark(label: str, since: float) -> None:
    if os.environ.get("PHOLOG_STARTUP_TIMING", "").lower() not in ("1", "true", "yes"):
        return
    print(f"[startup] {label}: {time.perf_counter() - since:.3f}s", flush=True)


@dataclass
class LiveConfig:
    model: str = "small"
    device: Optional[str] = None  # "cuda"|"cpu"|None(auto)
    compute_type: Optional[str] = None  # "float16"|"int8"|None(auto)
    language: Optional[str] = None
    beam_size: int = 1
    vad_filter: bool = True
    chunk_length_s: float = 15.0
    step_s: float = 2.0
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "float32"
    output_dir: Path = Path("./live_transcripts")
    session_name: Optional[str] = None
    write_audio_wav: bool = True
    lsl: bool = False
    mic_device: Optional[str] = None  # name or index
    word_timestamps: bool = True
    no_speech_threshold: float = 0.6
    logprob_threshold: float = -1.0
    temperature: float = 0.0


class RingBuffer:
    def __init__(self, capacity_samples: int, dtype: np.dtype = np.float32):
        self.capacity = int(capacity_samples)
        self.buffer = np.zeros(self.capacity, dtype=dtype)
        self.write_pos = 0
        self.size = 0
        self.lock = threading.Lock()

    def append(self, data: np.ndarray):
        with self.lock:
            n = len(data)
            if n >= self.capacity:
                # keep only the last capacity samples
                self.buffer[:] = data[-self.capacity:]
                self.write_pos = 0
                self.size = self.capacity
                return
            end = self.write_pos + n
            if end <= self.capacity:
                self.buffer[self.write_pos:end] = data
            else:
                split = self.capacity - self.write_pos
                self.buffer[self.write_pos:] = data[:split]
                self.buffer[: end - self.capacity] = data[split:]
            self.write_pos = (self.write_pos + n) % self.capacity
            self.size = min(self.capacity, self.size + n)

    def get_last(self, n_samples: int) -> np.ndarray:
        with self.lock:
            n = min(n_samples, self.size)
            if n == 0:
                return np.zeros(0, dtype=self.buffer.dtype)
            start = (self.write_pos - n) % self.capacity
            if start + n <= self.capacity:
                return self.buffer[start : start + n].copy()
            else:
                split = self.capacity - start
                return np.concatenate((self.buffer[start:], self.buffer[: n - split])).copy()


class LiveTranscriber:
    def __init__(self, cfg: LiveConfig):
        self.cfg = cfg
        # Resolve auto (None) without mutating cfg so GUI settings keep device/compute_type as None
        self._resolved_device = cfg.device or ("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        self._resolved_compute_type = cfg.compute_type or ("float16" if self._resolved_device == "cuda" else "int8")

        _t_import = time.perf_counter()
        from faster_whisper import WhisperModel
        _startup_mark("import faster_whisper", _t_import)
        _t_model = time.perf_counter()
        self.model = WhisperModel(self.cfg.model, device=self._resolved_device, compute_type=self._resolved_compute_type)
        _startup_mark(f"WhisperModel({self.cfg.model!r}) construct", _t_model)

        self.recording_start_time = datetime.now()
        self._stop_event = threading.Event()
        self._audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=50)
        self._ring = RingBuffer(
            capacity_samples=int(self.cfg.sample_rate * max(self.cfg.chunk_length_s * 2, 60)),
            dtype=np.float32,
        )
        self._wav_file = None
        self._samples_written = 0
        self._samples_lock = threading.Lock()

        # Dedup state
        self._last_emitted_time: float = 0.0  # seconds since start

        # LSL
        self._lsl_outlet: Optional[StreamOutlet] = None
        if self.cfg.lsl and LSL_AVAILABLE:
            info = StreamInfo(
                name="transcript",
                type="Markers",
                channel_count=1,
                nominal_srate=0.0,
                channel_format="string",
                source_id=f"whisper_live_{int(time.time())}",
            )
            self._lsl_outlet = StreamOutlet(info)

        # Output paths
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        session = self.cfg.session_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.basepath = self.cfg.output_dir / session
        self.jsonl_path = self.basepath.with_suffix(".jsonl")
        self.wav_path = self.basepath.with_suffix(".wav")

    def _audio_callback(self, indata, frames, time_info, status):  # called by sounddevice thread
        if status:
            # Drop if queue full to avoid backpressure
            pass
        mono = indata[:, 0] if indata.ndim == 2 else indata
        try:
            self._audio_q.put_nowait(mono.copy())
        except queue.Full:
            # Drop the block; ring buffer continues via writer thread
            pass

    def _audio_writer(self):
        if self.cfg.write_audio_wav and sf is not None:
            self._wav_file = sf.SoundFile(
                self.wav_path.as_posix(), mode="w", samplerate=self.cfg.sample_rate, channels=1, subtype="PCM_16"
            )
        block_samples = 0
        while not self._stop_event.is_set():
            try:
                data = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
            self._ring.append(data.astype(np.float32, copy=False))
            if self._wav_file is not None:
                self._wav_file.write(data)
            block_samples += len(data)
            with self._samples_lock:
                self._samples_written += len(data)

        if self._wav_file is not None:
            self._wav_file.close()

    def _emit(self, segments: List[dict]):
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            for seg in segments:
                f.write(json.dumps(seg, ensure_ascii=False) + "\n")
        if self._lsl_outlet is not None:
            for seg in segments:
                try:
                    self._lsl_outlet.push_sample([seg.get("text", "")])
                except Exception:
                    pass

    def _relative_to_absolute(self, rel_sec: float) -> str:
        return (self.recording_start_time + timedelta(seconds=rel_sec)).isoformat()

    def _transcriber_loop(self):
        sample_rate = self.cfg.sample_rate
        window = int(self.cfg.chunk_length_s * sample_rate)
        step = max(1, int(self.cfg.step_s * sample_rate))
        last_run_samples = 0
        # Pace the loop based on step_s
        next_time = time.time()
        while not self._stop_event.is_set():
            now = time.time()
            if now < next_time:
                time.sleep(min(0.02, next_time - now))
                continue
            next_time = now + self.cfg.step_s

            # Snapshot samples written to compute absolute offset
            with self._samples_lock:
                samples_written = self._samples_written
            audio = self._ring.get_last(window)
            if len(audio) < window // 2:  # wait for enough context
                continue
            # Compute absolute offset of the start of this window
            window_start_abs_sec = max(0.0, (samples_written - len(audio)) / sample_rate)

            # Run inference on the window tail
            # We want word timestamps for robust dedup
            try:
                segments, info = self.model.transcribe(
                    audio,
                    language=self.cfg.language,
                    beam_size=self.cfg.beam_size,
                    vad_filter=self.cfg.vad_filter,
                    temperature=self.cfg.temperature,
                    word_timestamps=self.cfg.word_timestamps,
                    no_speech_threshold=self.cfg.no_speech_threshold,
                    condition_on_previous_text=True,
                )
            except TypeError as e:
                # Fallback for older faster-whisper versions that don't support some kwargs
                try:
                    segments, info = self.model.transcribe(
                        audio,
                        language=self.cfg.language,
                        beam_size=self.cfg.beam_size,
                        vad_filter=self.cfg.vad_filter,
                        temperature=self.cfg.temperature,
                        word_timestamps=self.cfg.word_timestamps,
                    )
                except Exception:
                    continue
            except Exception:
                continue

            new_emissions: List[dict] = []
            for seg in segments:  # seg is faster_whisper.transcribe.Segment
                # faster-whisper segment times are relative to provided audio
                seg_start = window_start_abs_sec + max(0.0, float(seg.start))
                seg_end = window_start_abs_sec + max(0.0, float(seg.end))

                new_words = []
                max_new_end = self._last_emitted_time
                if getattr(seg, "words", None):
                    for w in seg.words:
                        w_start = window_start_abs_sec + float(w.start)
                        w_end = window_start_abs_sec + float(w.end)
                        if w_end <= self._last_emitted_time + 0.02:
                            continue
                        new_words.append(
                            {
                                "text": w.word,
                                "start": w_start,
                                "end": w_end,
                                "probability": getattr(w, "probability", None),
                            }
                        )
                        if w_end > max_new_end:
                            max_new_end = w_end
                else:
                    # Fallback to segment-level if no words available
                    if seg_end <= self._last_emitted_time + 0.02:
                        continue
                    new_words.append({"text": seg.text.strip(), "start": seg_start, "end": seg_end})
                    max_new_end = max(max_new_end, seg_end)

                if not new_words:
                    continue

                text = " ".join([w["text"] for w in new_words]).strip()
                abs_start = self._relative_to_absolute(new_words[0]["start"])
                abs_end = self._relative_to_absolute(new_words[-1]["end"])
                emission = {
                    "text": text,
                    "start": new_words[0]["start"],
                    "end": new_words[-1]["end"],
                    "absolute_start": abs_start,
                    "absolute_end": abs_end,
                    "words": new_words,
                }
                new_emissions.append(emission)
                self._last_emitted_time = max_new_end

            if new_emissions:
                self._emit(new_emissions)

    def start(self):
        if sd is None:
            raise RuntimeError("sounddevice/soundfile not available. Please install extras.")

        # Start audio capture
        self._stop_event.clear()
        self._writer_thread = threading.Thread(target=self._audio_writer, daemon=True)
        self._writer_thread.start()

        self._stream = sd.InputStream(
            samplerate=self.cfg.sample_rate,
            channels=self.cfg.channels,
            dtype=self.cfg.dtype,
            device=self.cfg.mic_device,
            blocksize=int(self.cfg.sample_rate * 0.05),  # ~50ms blocks
            callback=self._audio_callback,
        )
        self._stream.start()

        # Start inference loop
        self._transcribe_thread = threading.Thread(target=self._transcriber_loop, daemon=True)
        self._transcribe_thread.start()

    def stop(self):
        self._stop_event.set()
        try:
            if hasattr(self, "_stream"):
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        if hasattr(self, "_transcribe_thread"):
            self._transcribe_thread.join(timeout=2.0)
        if hasattr(self, "_writer_thread"):
            self._writer_thread.join(timeout=2.0)


def cli(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="Near real-time microphone transcription with faster-whisper")
    p.add_argument("--model", default="small", help="Model size or path (e.g., tiny, base, small, medium, large-v3)")
    p.add_argument("--device", default=None, help="cuda or cpu (auto if omitted)")
    p.add_argument("--compute-type", dest="compute_type", default=None, help="float16, int8, int8_float16, etc. (auto if omitted)")
    p.add_argument("--language", default=None, help="Hint language code (auto-detect if omitted)")
    p.add_argument("--beam-size", dest="beam_size", type=int, default=1)
    p.add_argument("--no-vad", dest="no_vad", action="store_true", help="Disable internal VAD filter")
    p.add_argument("--chunk-length", dest="chunk_length_s", type=float, default=15.0, help="Inference window seconds")
    p.add_argument("--step", dest="step_s", type=float, default=2.0, help="Step seconds between inferences")
    p.add_argument("--sr", dest="sample_rate", type=int, default=16000)
    p.add_argument("--output-dir", dest="output_dir", default="./live_transcripts")
    p.add_argument("--session", dest="session_name", default=None)
    p.add_argument("--no-wav", dest="no_wav", action="store_true", help="Do not save WAV recording")
    p.add_argument("--lsl", dest="lsl", action="store_true", help="Emit LSL Markers stream of segments")
    p.add_argument("--mic-device", dest="mic_device", default=None, help="Input device name or index")
    p.add_argument("--no-word-timestamps", dest="no_word_timestamps", action="store_true", help="Disable word-level timestamps (enabled by default)")
    p.add_argument("--temperature", dest="temperature", type=float, default=0.0)
    p.add_argument("--no-speech-threshold", dest="no_speech_threshold", type=float, default=0.6)
    p.add_argument("--logprob-threshold", dest="logprob_threshold", type=float, default=-1.0)

    args = p.parse_args(argv)

    cfg = LiveConfig(
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        beam_size=args.beam_size,
        vad_filter=not args.no_vad,
        chunk_length_s=args.chunk_length_s,
        step_s=args.step_s,
        sample_rate=args.sample_rate,
        output_dir=Path(args.output_dir),
        session_name=args.session_name,
        write_audio_wav=not args.no_wav,
        lsl=args.lsl,
        mic_device=args.mic_device,
        word_timestamps=not args.no_word_timestamps,
        temperature=args.temperature,
        no_speech_threshold=args.no_speech_threshold,
        logprob_threshold=args.logprob_threshold,
    )

    lt = LiveTranscriber(cfg)
    print(
        f"Starting live transcription -> JSONL: {lt.jsonl_path.as_posix()} WAV: {lt.wav_path.as_posix() if cfg.write_audio_wav else '(disabled)'} | device={lt._resolved_device} compute_type={lt._resolved_compute_type}"
    )
    try:
        lt.start()
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        lt.stop()
    print("Stopped.")


if __name__ == "__main__":
    cli()


