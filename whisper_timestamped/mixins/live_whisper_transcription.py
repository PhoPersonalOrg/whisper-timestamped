from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from copy import deepcopy
from RealtimeSTT import AudioToTextRecorder
import pyautogui
import pylsl
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import pyxdf
from datetime import datetime, timedelta
import os
import threading
import time
import numpy as np
import json
import pickle
import mne
from pathlib import Path
import pystray
from PIL import Image, ImageDraw
import keyboard
import pyautogui
import socket
import sys
import logging

from phopylslhelper.general_helpers import unwrap_single_element_listlike_if_needed, readable_dt_str, from_readable_dt_str, localize_datetime_to_timezone, tz_UTC, tz_Eastern, _default_tz
from phopylslhelper.easy_time_sync import EasyTimeSyncParsingMixin

# Import the live transcription components
from whisper_timestamped.live import LiveTranscriber, LiveConfig
try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    sd = None

import os  # add at top if not present
program_lock_port = int(os.environ.get("LIVE_WHISPER_LOCK_PORT", 13371))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


_default_xdf_folder = Path(r'E:/Dropbox (Personal)/Databases/UnparsedData/PhoLogToLabStreamingLayer_logs').resolve()
# _default_xdf_folder = Path('/media/halechr/MAX/cloud/University of Michigan Dropbox/Pho Hale/Personal/LabRecordedTextLog').resolve() ## Lab computer

whisper_audio_out_dir: Path = Path("L:/ScreenRecordings/EyeTrackerVR_Recordings/audio").resolve()
whisper_transcripts_dir: Path = Path("L:/ScreenRecordings/EyeTrackerVR_Recordings/transcriptions").resolve()
whisper_lsl_converted_transcripts_dir: Path = whisper_transcripts_dir.joinpath('LSL_Converted').resolve()
whisper_live_transcripts_dir: Path = Path("E:/Dropbox (Personal)/Databases/UnparsedData/PhoLogToLabStreamingLayer_logs/live_transcripts").resolve()


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
        self.start_transcription_button = ttk.Button(transcription_frame, text="Start Transcription",
                                                   command=self.start_live_transcription)
        self.start_transcription_button.grid(row=0, column=1, padx=5)

        self.stop_transcription_button = ttk.Button(transcription_frame, text="Stop Transcription",
                                                  command=self.stop_live_transcription, state="disabled")
        self.stop_transcription_button.grid(row=0, column=2, padx=5)

        self.transcription_settings_button = ttk.Button(transcription_frame, text="Settings",
                                                       command=self.show_transcription_settings)
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

            # # Update LSL status label safely
            # try:
            #     if not self._shutting_down:
            #         self.lsl_status_label.config(text="LSL Status: Connected", foreground="green")
            # except tk.TclError:
            #     pass  # GUI is being destroyed

            # # Setup inlet for recording our own stream (with delay to allow outlet to be discovered)
            # self.root.after(1000, self.setup_recording_inlet)

        except Exception as e:
            print(f"Error creating WhisperLiveLogger LSL outlet: {e}")
            self.outlets['WhisperLiveLogger'] = None
            # try:
            #     if not self._shutting_down:
            #         self.lsl_status_label.config(text=f"LSL Status: Error - {str(e)}", foreground="red")
            # except tk.TclError:
            #     pass  # GUI is being destroyed
            
            raise

    # ---------------------------------------------------------------------------- #
    #                           Live Transcription Methods                         #
    # ---------------------------------------------------------------------------- #

    def setup_transcription_config(self):
        """Setup default transcription configuration

        captures: whisper_live_transcripts_dir

        """
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
            output_dir=whisper_live_transcripts_dir,
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


    def start_live_transcription(self):
        """Start live audio transcription"""
        if not AUDIO_AVAILABLE:
            messagebox.showerror("Error", "Audio libraries not available. Please install sounddevice and soundfile.")
            return

        logger.info(f".start_live_transcription()  hit")

        if self.transcription_active:
            return

        try:
            # Setup configuration if not already done
            if not self.transcription_config:
                self.setup_transcription_config()

            # Set session name with timestamp
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.transcription_config.session_name = session_name

            # Set selected audio device
            selected_device = self.audio_device_var.get()
            if selected_device != "Default":
                # Extract device index from the selection
                device_index = int(selected_device.split(":")[0])
                self.transcription_config.mic_device = device_index

            # Create transcriber instance
            self.live_transcriber = LiveTranscriber(self.transcription_config)
            logger.info(f"\t created live transcriber instance.")

            # Override the _emit method to send to our LSL stream
            original_emit = self.live_transcriber._emit
            def custom_emit(segments):
                # Send to our LSL outlet
                logger.info(f".custom_emit(segments={segments})  hit")
                for seg in segments:
                    text = seg.get("text", "").strip()
                    if text:
                        self.send_lsl_message(text)
                        # Update GUI display
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.update_log_display(f"[TRANSCRIBED] {text}", timestamp)

                # Also call original emit for file logging
                original_emit(segments)

            self.live_transcriber._emit = custom_emit

            # Start transcription
            self.live_transcriber.start()
            logger.info(f"\t started live transcription.")
            
            self.transcription_active = True

            # Update GUI
            try:
                if not self._shutting_down:
                    self.transcription_status_label.config(text="Transcribing...", foreground="green")
                    self.start_transcription_button.config(state="disabled")
                    self.stop_transcription_button.config(state="normal")
                    self.transcription_settings_button.config(state="disabled")
            except tk.TclError:
                pass

            self.update_log_display("Live transcription started", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print(f"Live transcription started with session: {session_name}")

        except Exception as e:
            logger.error(f".start_live_transcription()  error: {e}")
            messagebox.showerror("Error", f"Failed to start live transcription: {str(e)}")
            print(f"Error starting transcription: {e}")
            import traceback
            traceback.print_exc()


    def stop_live_transcription(self):
        """Stop live audio transcription"""
        logger.info(f".stop_live_transcription()  hit")
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
            self.start_live_transcription()
        except Exception as e:
            print(f'auto_start_live_transcription(): encountered error {e}.')
            raise



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
        model_combo = ttk.Combobox(main_frame, textvariable=model_var,
                                  values=["tiny", "base", "small", "medium", "large-v3"],
                                  state="readonly")
        model_combo.pack(fill=tk.X, pady=(0, 10))

        # Language selection
        ttk.Label(main_frame, text="Language (leave empty for auto-detect):").pack(anchor=tk.W, pady=(0, 5))
        language_var = tk.StringVar(value=self.transcription_config.language or "")
        language_entry = ttk.Entry(main_frame, textvariable=language_var)
        language_entry.pack(fill=tk.X, pady=(0, 10))

        # Device selection
        ttk.Label(main_frame, text="Processing Device:").pack(anchor=tk.W, pady=(0, 5))
        device_var = tk.StringVar(value=self.transcription_config.device or "auto")
        device_combo = ttk.Combobox(main_frame, textvariable=device_var,
                                   values=["auto", "cpu", "cuda"], state="readonly")
        device_combo.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(main_frame, text="Auto uses CUDA when available, otherwise CPU.", font=("TkDefaultFont", 8)).pack(anchor=tk.W, pady=(0, 10))

        # VAD filter
        vad_var = tk.BooleanVar(value=self.transcription_config.vad_filter)
        ttk.Checkbutton(main_frame, text="Enable Voice Activity Detection (VAD)",
                       variable=vad_var).pack(anchor=tk.W, pady=(0, 10))

        # Chunk length
        ttk.Label(main_frame, text="Chunk Length (seconds):").pack(anchor=tk.W, pady=(0, 5))
        chunk_var = tk.DoubleVar(value=self.transcription_config.chunk_length_s)
        chunk_spin = ttk.Spinbox(main_frame, from_=5.0, to=30.0, increment=1.0,
                                textvariable=chunk_var, format="%.1f")
        chunk_spin.pack(fill=tk.X, pady=(0, 10))

        # Step size
        ttk.Label(main_frame, text="Step Size (seconds):").pack(anchor=tk.W, pady=(0, 5))
        step_var = tk.DoubleVar(value=self.transcription_config.step_s)
        step_spin = ttk.Spinbox(main_frame, from_=0.5, to=10.0, increment=0.5,
                               textvariable=step_var, format="%.1f")
        step_spin.pack(fill=tk.X, pady=(0, 10))

        # Save audio
        save_audio_var = tk.BooleanVar(value=self.transcription_config.write_audio_wav)
        ttk.Checkbutton(main_frame, text="Save audio to WAV file",
                       variable=save_audio_var).pack(anchor=tk.W, pady=(0, 10))

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

