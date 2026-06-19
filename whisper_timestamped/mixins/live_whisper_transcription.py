from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from pathlib import Path
import pylsl
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from datetime import datetime, timedelta
import os
import threading
import time
import logging

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    sd = None

logger = logging.getLogger(__name__)

_WHISPER_LIVE_TRANSCRIPTS_DIR_RAW = Path(r"E:/Dropbox (Personal)/Databases/UnparsedData/PhoLogToLabStreamingLayer_logs/live_transcripts")
_resolved_whisper_live_transcripts_dir: Optional[Path] = None


def get_whisper_live_transcripts_dir() -> Path:
    global _resolved_whisper_live_transcripts_dir
    if _resolved_whisper_live_transcripts_dir is None:
        _resolved_whisper_live_transcripts_dir = _WHISPER_LIVE_TRANSCRIPTS_DIR_RAW.resolve()
    return _resolved_whisper_live_transcripts_dir


def _import_live_transcription():
    from whisper_timestamped.live import LiveTranscriber, LiveConfig
    return LiveTranscriber, LiveConfig


class LiveWhisperTranscriptionAppMixin:
    """ 

    self.init_LiveWhisperTranscriptionAppMixin()
    self.setup_LiveWhisperTranscriptionAppMixin_lsl_outlet()
    self.setup_gui_LiveWhisperTranscriptionAppMixin(main_frame)

    Usage:
        from whisper_timestamped.mixins.live_whisper_transcription import LiveWhisperTranscriptionAppMixin

    """
    @property
    def outlet_LiveWhisperTranscriptionAppMixin(self) -> Optional[pylsl.StreamOutlet]:
        """The outlet_LiveWhisperTranscriptionAppMixin property."""
        return self.outlets['WhisperLiveLogger']
    @outlet_LiveWhisperTranscriptionAppMixin.setter
    def outlet_LiveWhisperTranscriptionAppMixin(self, value):
        self.outlets['WhisperLiveLogger'] = value


    def init_LiveWhisperTranscriptionAppMixin(self):
        # Live transcription state
        self.live_transcriber = None
        self.whisper_live_transcript_path = None
        self.transcription_active = False
        self.transcription_config = None
        self._transcription_loading = False
        self._transcription_cancel_requested = False
        self._transcription_start_thread: Optional[threading.Thread] = None
        self._pending_transcriber = None


    def setup_LiveWhisperTranscriptionAppMixin(self):
        # Setup transcription configuration
        self.setup_transcription_config()


    def setup_gui_LiveWhisperTranscriptionAppMixin(self, main_frame: ttk.Frame, row: int=2):
        """Create the GUI elements"""
        # Live Transcription control frame
        transcription_frame = ttk.LabelFrame(main_frame, text="Live Audio Transcription", padding="5")
        transcription_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        transcription_frame.columnconfigure(2, weight=1)

        # Transcription status
        self.transcription_status_label = ttk.Label(transcription_frame, text="Not Transcribing", foreground="red")
        self.transcription_status_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 10))

        # Transcription buttons
        self.start_transcription_button = ttk.Button(transcription_frame, text="Start Transcription", command=self.start_live_transcription)
        self.start_transcription_button.grid(row=0, column=1, padx=5)

        self.stop_transcription_button = ttk.Button(transcription_frame, text="Stop Transcription", command=self.stop_live_transcription, state="disabled")
        self.stop_transcription_button.grid(row=0, column=2, padx=5)

        self.transcription_settings_button = ttk.Button(transcription_frame, text="Settings", command=self.show_transcription_settings)
        self.transcription_settings_button.grid(row=0, column=3, padx=5)

        # Audio device selection
        ttk.Label(transcription_frame, text="Audio Device:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))

        self.audio_device_var = tk.StringVar(value="Default")
        self.audio_device_combo = ttk.Combobox(transcription_frame, textvariable=self.audio_device_var, state="readonly", width=30)
        self.audio_device_combo.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=(5, 0))

        # Populate audio devices
        self.refresh_audio_devices()

        # Refresh devices button
        ttk.Button(transcription_frame, text="Refresh", command=self.refresh_audio_devices).grid(row=1, column=3, padx=5, pady=(5, 0))
        return transcription_frame




    # ==================================================================================================================================================================================================================================================================================== #
    # General Methods                                                                                                                                                                                                                                                                      #
    # ==================================================================================================================================================================================================================================================================================== #
    def setup_lsl_outlet_LiveWhisperTranscriptionAppMixin(self):
        """called from `self.setup_lsl_outlet()
        Create an LSL outlet for sending messages

        sets up `self.outlet_LiveWhisperTranscriptionAppMixin`

        """
        assert self.outlets is not None
        try:
            
            # Create stream info
            info = pylsl.StreamInfo(
                name='WhisperLiveLogger',
                type='Markers',
                channel_count=1,
                nominal_srate=pylsl.IRREGULAR_RATE,
                channel_format=pylsl.cf_string,
                source_id='textlogger_002'
            )

            # Add some metadata
            info.desc().append_child_value("manufacturer", "PhoWhisperTimestampedLive")
            info.desc().append_child_value("version", "1.0")
            info.desc().append_child_value("description", "Live transcribed audio")
            info.desc().append_child_value('hostname', 'TODO')

            ## add a custom timestamp field to the stream info:
            info = self.EasyTimeSyncParsingMixin_add_lsl_outlet_info(info=info)


            # Create outlet
            self.outlets['WhisperLiveLogger'] = pylsl.StreamOutlet(info)
            print("WhisperLiveLogger LSL outlet created successfully")

        except Exception as e:
            print(f"Error creating WhisperLiveLogger LSL outlet: {e}")
            self.outlets['WhisperLiveLogger'] = None
            
            raise

    # ---------------------------------------------------------------------------- #
    #                           Live Transcription Methods                         #
    # ---------------------------------------------------------------------------- #

    def setup_transcription_config(self):
        """Setup default transcription configuration

        captures: whisper_live_transcripts_dir

        """
        _, LiveConfig = _import_live_transcription()
        self.transcription_config = LiveConfig(
            model="medium",  # Good balance of speed and accuracy
            device=None,  # Auto-detect: LiveTranscriber prefers CUDA when available, falls back to CPU
            compute_type=None,  # Auto-detect: float16 on CUDA, int8 on CPU
            language='en',  # Auto-detect
            beam_size=1,
            vad_filter=True,
            chunk_length_s=15.0,
            step_s=2.0,
            sample_rate=16000,
            channels=1,
            dtype="float32",
            output_dir=get_whisper_live_transcripts_dir(),
            session_name=None,
            write_audio_wav=True,
            lsl=False,  # We'll handle LSL ourselves
            mic_device=None,
            word_timestamps=True,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            temperature=0.0
        )


    @property
    def whisper_live_transcripts_dir(self) -> Path:
        """The whisper_live_transcripts_dir property."""
        return self.transcription_config.output_dir
    @whisper_live_transcripts_dir.setter
    def whisper_live_transcripts_dir(self, value: Path):
        self.transcription_config.output_dir = value


    def get_audio_devices(self):
        """Get list of available audio input devices"""
        if not AUDIO_AVAILABLE:
            return []

        try:
            devices = sd.query_devices()
            input_devices = []
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    input_devices.append((i, device['name']))
            return input_devices
        except Exception as e:
            print(f"Error getting audio devices: {e}")
            return []


    def _set_transcription_ui_loading(self):
        def _update():
            if getattr(self, '_shutting_down', False):
                return
            try:
                self.transcription_status_label.config(text="Loading transcription model...", foreground="orange")
                self.start_transcription_button.config(state="disabled")
                self.stop_transcription_button.config(state="disabled")
                self.transcription_settings_button.config(state="disabled")
            except tk.TclError:
                pass
        self.root.after(0, _update)


    def _apply_transcription_started_ui(self, session_name: str):
        if getattr(self, '_shutting_down', False):
            return
        try:
            self.transcription_status_label.config(text="Transcribing...", foreground="green")
            self.start_transcription_button.config(state="disabled")
            self.stop_transcription_button.config(state="normal")
            self.transcription_settings_button.config(state="disabled")
        except tk.TclError:
            pass
        self.update_log_display("Live transcription started", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(f"Live transcription started with session: {session_name}")


    def _apply_transcription_start_failed_ui(self, error: Exception, silent: bool):
        if getattr(self, '_shutting_down', False):
            return
        try:
            self.transcription_status_label.config(text="Not Transcribing", foreground="red")
            self.start_transcription_button.config(state="normal")
            self.stop_transcription_button.config(state="disabled")
            self.transcription_settings_button.config(state="normal")
        except tk.TclError:
            pass
        if not silent:
            messagebox.showerror("Error", f"Failed to start live transcription: {str(error)}")


    def _push_whisper_lsl_message(self, message: str):
        """Push a transcribed text segment to the WhisperLiveLogger outlet, falling back to TextLogger."""
        whisper_outlet = None
        try:
            whisper_outlet = (self.outlets or {}).get('WhisperLiveLogger')
        except Exception:
            whisper_outlet = None
        if whisper_outlet is not None:
            try:
                whisper_outlet.push_sample([message])
                return
            except Exception as e:
                print(f"Error pushing WhisperLiveLogger sample: {e}")
        # Fallback: route through TextLogger via send_lsl_message
        try:
            self.send_lsl_message(message)
        except Exception as e:
            print(f"Error sending fallback LSL message: {e}")


    def _make_custom_emit(self, original_emit):
        """Build a thread-safe emit callback. Runs on the inference thread; marshals UI/LSL to the main thread."""
        def custom_emit(segments):
            try:
                logger.info(f".custom_emit(segments_count={len(segments)})")
            except Exception:
                pass
            # Snapshot text fragments so the main-thread closure doesn't depend on segment lifetimes
            texts = [seg.get("text", "").strip() for seg in segments]
            texts = [t for t in texts if t]
            if texts:
                def _ui_update():
                    if getattr(self, '_shutting_down', False):
                        return
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    for text in texts:
                        self._push_whisper_lsl_message(text)
                        try:
                            self.update_log_display(f"[TRANSCRIBED] {text}", timestamp)
                        except Exception as e:
                            print(f"Error updating log display from emit: {e}")
                try:
                    self.root.after(0, _ui_update)
                except Exception as e:
                    print(f"Error scheduling emit UI update: {e}")
            # File logging stays on the inference thread
            try:
                original_emit(segments)
            except Exception as e:
                print(f"Error in original whisper emit: {e}")
        return custom_emit


    def _start_live_transcription_worker(self, silent: bool, session_name: str, mic_device):
        """Background-thread worker that loads the Whisper model only. Mic capture is started later on the main thread."""
        try:
            LiveTranscriber, _ = _import_live_transcription()
            logger.info(".start_live_transcription() loading model in background thread")
            transcriber = LiveTranscriber(self.transcription_config)
            logger.info("\t created live transcriber instance.")

            self._pending_transcriber = transcriber
            self.root.after(0, lambda: self._complete_transcription_start(transcriber, session_name, silent))
        except Exception as e:
            logger.error(f".start_live_transcription() model load error: {e}")
            print(f"Error loading transcription model: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda err=e: self._finalize_transcription_start_failure(err, silent))


    def _complete_transcription_start(self, transcriber, session_name: str, silent: bool):
        """Main-thread completion: wire emit, start sounddevice, update UI."""
        # User cancelled (Stop pressed) or app is shutting down while the model loaded
        if self._transcription_cancel_requested or getattr(self, '_shutting_down', False):
            logger.info("Transcription start cancelled before mic could open; discarding loaded model.")
            try:
                transcriber.stop()
            except Exception:
                pass
            self._pending_transcriber = None
            self._transcription_cancel_requested = False
            self._transcription_loading = False
            self._transcription_start_thread = None
            self._apply_transcription_start_failed_ui(RuntimeError("cancelled"), silent=True)
            return

        try:
            original_emit = transcriber._emit
            transcriber._emit = self._make_custom_emit(original_emit)
            transcriber.start()
            logger.info("\t started live transcription (mic + inference loop running).")

            self.live_transcriber = transcriber
            self.transcription_active = True
            self._pending_transcriber = None
            self._apply_transcription_started_ui(session_name)
        except Exception as e:
            logger.error(f".start_live_transcription() mic start error: {e}")
            print(f"Error starting transcription mic: {e}")
            import traceback
            traceback.print_exc()
            try:
                transcriber.stop()
            except Exception:
                pass
            self._pending_transcriber = None
            self._apply_transcription_start_failed_ui(e, silent)
        finally:
            self._transcription_loading = False
            self._transcription_cancel_requested = False
            self._transcription_start_thread = None


    def _finalize_transcription_start_failure(self, error: Exception, silent: bool):
        """Main-thread cleanup after a background-load failure."""
        self._pending_transcriber = None
        self._transcription_loading = False
        self._transcription_cancel_requested = False
        self._transcription_start_thread = None
        self._apply_transcription_start_failed_ui(error, silent)


    def start_live_transcription(self, silent: bool = False):
        """Start live audio transcription"""
        if not AUDIO_AVAILABLE:
            msg = "Audio libraries not available. Please install sounddevice and soundfile."
            if silent:
                print(f"start_live_transcription(): {msg}")
            else:
                messagebox.showerror("Error", msg)
            return

        logger.info(f".start_live_transcription()  hit")

        if self.transcription_active or self._transcription_loading:
            return

        if not self.transcription_config:
            self.setup_transcription_config()

        session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.transcription_config.session_name = session_name

        selected_device = self.audio_device_var.get()
        if selected_device != "Default":
            device_index = int(selected_device.split(":")[0])
            self.transcription_config.mic_device = device_index
        else:
            self.transcription_config.mic_device = None

        self._transcription_loading = True
        self._transcription_cancel_requested = False
        self._set_transcription_ui_loading()
        self._transcription_start_thread = threading.Thread(target=self._start_live_transcription_worker, args=(silent, session_name, self.transcription_config.mic_device), daemon=True)
        self._transcription_start_thread.start()


    def stop_live_transcription(self):
        """Stop live audio transcription. Also cancels an in-progress model load."""
        logger.info(f".stop_live_transcription()  hit")

        # Case 1: model is still loading on the background thread
        if self._transcription_loading and not self.transcription_active:
            logger.info("stop requested while model is still loading; will cancel on completion")
            self._transcription_cancel_requested = True
            # Reflect intent immediately in the UI; the worker will finalize cleanup
            try:
                if not self._shutting_down:
                    self.transcription_status_label.config(text="Cancelling...", foreground="red")
                    self.stop_transcription_button.config(state="disabled")
            except tk.TclError:
                pass
            return

        if not self.transcription_active:
            return

        try:
            if self.live_transcriber:
                self.live_transcriber.stop()
                self.live_transcriber = None

            self.transcription_active = False

            # Update GUI
            try:
                if not self._shutting_down:
                    self.transcription_status_label.config(text="Not Transcribing", foreground="red")
                    self.start_transcription_button.config(state="normal")
                    self.stop_transcription_button.config(state="disabled")
                    self.transcription_settings_button.config(state="normal")
            except tk.TclError:
                pass

            self.update_log_display("Live transcription stopped", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print("Live transcription stopped")

        except Exception as e:
            print(f"Error stopping transcription: {e}")


    def auto_start_live_transcription(self):
        """ tries to start live transcription on startup """
        try:
            self.start_live_transcription(silent=True)
        except Exception as e:
            print(f'auto_start_live_transcription(): encountered error {e}.')



    def show_transcription_settings(self):
        """Show transcription settings dialog"""
        if not self.transcription_config:
            self.setup_transcription_config()

        settings_window = tk.Toplevel(self.root)
        settings_window.title("Transcription Settings")
        settings_window.geometry("400x500")
        settings_window.transient(self.root)
        settings_window.grab_set()

        # Center the window
        settings_window.update_idletasks()
        x = (settings_window.winfo_screenwidth() // 2) - (400 // 2)
        y = (settings_window.winfo_screenheight() // 2) - (500 // 2)
        settings_window.geometry(f"+{x}+{y}")

        main_frame = ttk.Frame(settings_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Model selection
        ttk.Label(main_frame, text="Whisper Model:").pack(anchor=tk.W, pady=(0, 5))
        model_var = tk.StringVar(value=self.transcription_config.model)
        model_combo = ttk.Combobox(main_frame, textvariable=model_var, values=["tiny", "base", "small", "medium", "large-v3"], state="readonly")
        model_combo.pack(fill=tk.X, pady=(0, 10))

        # Language selection
        ttk.Label(main_frame, text="Language (leave empty for auto-detect):").pack(anchor=tk.W, pady=(0, 5))
        language_var = tk.StringVar(value=self.transcription_config.language or "")
        language_entry = ttk.Entry(main_frame, textvariable=language_var)
        language_entry.pack(fill=tk.X, pady=(0, 10))

        # Device selection
        ttk.Label(main_frame, text="Processing Device:").pack(anchor=tk.W, pady=(0, 5))
        device_var = tk.StringVar(value=self.transcription_config.device or "auto")
        device_combo = ttk.Combobox(main_frame, textvariable=device_var, values=["auto", "cpu", "cuda"], state="readonly")
        device_combo.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(main_frame, text="Auto uses CUDA when available, otherwise CPU.", font=("TkDefaultFont", 8)).pack(anchor=tk.W, pady=(0, 10))

        # VAD filter
        vad_var = tk.BooleanVar(value=self.transcription_config.vad_filter)
        ttk.Checkbutton(main_frame, text="Enable Voice Activity Detection (VAD)", variable=vad_var).pack(anchor=tk.W, pady=(0, 10))

        # Chunk length
        ttk.Label(main_frame, text="Chunk Length (seconds):").pack(anchor=tk.W, pady=(0, 5))
        chunk_var = tk.DoubleVar(value=self.transcription_config.chunk_length_s)
        chunk_spin = ttk.Spinbox(main_frame, from_=5.0, to=30.0, increment=1.0, textvariable=chunk_var, format="%.1f")
        chunk_spin.pack(fill=tk.X, pady=(0, 10))

        # Step size
        ttk.Label(main_frame, text="Step Size (seconds):").pack(anchor=tk.W, pady=(0, 5))
        step_var = tk.DoubleVar(value=self.transcription_config.step_s)
        step_spin = ttk.Spinbox(main_frame, from_=0.5, to=10.0, increment=0.5, textvariable=step_var, format="%.1f")
        step_spin.pack(fill=tk.X, pady=(0, 10))

        # Save audio
        save_audio_var = tk.BooleanVar(value=self.transcription_config.write_audio_wav)
        ttk.Checkbutton(main_frame, text="Save audio to WAV file", variable=save_audio_var).pack(anchor=tk.W, pady=(0, 10))

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))

        def save_settings():
            self.transcription_config.model = model_var.get()
            self.transcription_config.language = language_var.get() or None
            self.transcription_config.device = device_var.get() if device_var.get() != "auto" else None
            self.transcription_config.vad_filter = vad_var.get()
            self.transcription_config.chunk_length_s = chunk_var.get()
            self.transcription_config.step_s = step_var.get()
            self.transcription_config.write_audio_wav = save_audio_var.get()
            settings_window.destroy()

        ttk.Button(button_frame, text="Save", command=save_settings).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="Cancel", command=settings_window.destroy).pack(side=tk.RIGHT)


    def refresh_audio_devices(self):
        """Refresh the list of available audio devices"""
        devices = self.get_audio_devices()
        device_list = ["Default"]

        for device_id, device_name in devices:
            device_list.append(f"{device_id}: {device_name}")

        self.audio_device_combo['values'] = device_list

        # Set to default if current selection is not in the list
        if self.audio_device_var.get() not in device_list:
            self.audio_device_var.set("Default")
