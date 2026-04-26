import os
# import sys
# import argparse
import json
import time
from pathlib import Path
from typing import List

import torch
from whisper.utils import str2bool, optional_float, optional_int
import whisper_timestamped as whisper
from whisper_timestamped.transcribe import write_csv, flatten, remove_keys, get_vad_segments
from whisper_timestamped.parse_video_filename import build_EDF_compatible_video_filename, parse_video_filename
# from whisper_timestamped import remove_non_speech
from whisper_timestamped.transcribe import remove_non_speech

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
    


def find_extant_output_files(output_dir: Path, base_name: str, output_formats = ['json', 'csv', 'srt', 'vtt', 'txt']) -> List[Path]:
    """ found any of the output files that would be created upon transcode completion in the output_dir 

    found_output_files: List[Path] = find_extant_output_files(output_dir=output_dir, base_name=base_name, output_formats=output_formats)

    """
    # Generate output filenames
    output_file_path: Path = output_dir.joinpath(base_name) ## with no suffix
    found_output_files: List[Path] = [] #{'json': {}, 'srt': {}, 'csv': {}}
    for k in output_formats:
        a_file: Path = output_file_path.with_suffix(f".{k}")
        if (a_file.exists() and a_file.is_file()):
            found_output_files.append(a_file)

    return found_output_files

def write_results(result, output_dir: Path, base_name: str, output_formats = ['json', 'csv', 'srt', 'vtt', 'txt']):
    """ Writes the results object out to disk
    base_name = video_file.stem
    output_files = write_results(result, output_dir=output_dir, base_name=base_name)

    """
    # Generate output filenames
    output_file_path: Path = output_dir.joinpath(base_name) ## with no suffix
    print(F'building output files for output_file_path: "{output_file_path.as_posix()}"')
    output_files = {k:dict() for k in output_formats} #{'json': {}, 'srt': {}, 'csv': {}}

    ## Save JSON:
    if "json" in output_formats:
        try:
            # save JSON
            a_file = output_file_path.with_suffix(".words.json")
            with open(a_file, "w", encoding="utf-8") as js:
                json.dump(result, js, indent=2, ensure_ascii=False)
            output_files['.'.join([k.removeprefix('.') for k in a_file.suffixes])][base_name] = a_file
            print(f"  ✓ Saved: {a_file.name}")
        except Exception as e:
            print(f"  ✗ Error saving JSON: {str(e)}")

    # save CSV
    if "csv" in output_formats:
        try:
            a_file = output_file_path.with_suffix(".csv")
            with open(a_file, "w", encoding="utf-8") as csv:
                write_csv(result["segments"], file=csv, header=True)
            output_files['.'.join([k.removeprefix('.') for k in a_file.suffixes])][base_name] = a_file
            print(f"  ✓ Saved: {a_file.name}")
        except Exception as e:
            print(f"  ✗ Error saving CSV: {str(e)}")

        try:
            a_file = output_file_path.with_suffix(".words.csv")
            with open(a_file, "w", encoding="utf-8") as csv:
                write_csv(flatten(result["segments"], "words"), file=csv, header=True)
            output_files['.'.join([k.removeprefix('.') for k in a_file.suffixes])][base_name] = a_file
            print(f"  ✓ Saved: {a_file.name}")
        except Exception as e:
            print(f"  ✗ Error saving words CSV: {str(e)}")

    # save TXT
    if "txt" in output_formats:
        try:
            a_file = output_file_path.with_suffix(".txt")
            with open(a_file, "w", encoding="utf-8") as txt:
                write_txt(result["segments"], file=txt)
            output_files['.'.join([k.removeprefix('.') for k in a_file.suffixes])][base_name] = a_file
            print(f"  ✓ Saved: {a_file.name}")
        except Exception as e:
            print(f"  ✗ Error saving TXT: {str(e)}")

    # save VTT
    if "vtt" in output_formats:
        try:
            a_file = output_file_path.with_suffix(".vtt")
            with open(a_file, "w", encoding="utf-8") as vtt:
                write_vtt(remove_keys(result["segments"], "words"), file=vtt)
            output_files['.'.join([k.removeprefix('.') for k in a_file.suffixes])][base_name] = a_file
            print(f"  ✓ Saved: {a_file.name}")
        except Exception as e:
            print(f"  ✗ Error saving VTT: {str(e)}")

        try:
            a_file = output_file_path.with_suffix(".words.vtt")
            with open(a_file, "w", encoding="utf-8") as vtt:
                write_vtt(flatten(result["segments"], "words"), file=vtt)
            output_files['.'.join([k.removeprefix('.') for k in a_file.suffixes])][base_name] = a_file
            print(f"  ✓ Saved: {a_file.name}")
        except Exception as e:
            print(f"  ✗ Error saving words VTT: {str(e)}")

    # save SRT
    if "srt" in output_formats:
        try:
            a_file = output_file_path.with_suffix(".srt")
            with open(a_file, "w", encoding="utf-8") as srt:
                write_srt(remove_keys(result["segments"], "words"), file=srt)
            output_files['.'.join([k.removeprefix('.') for k in a_file.suffixes])][base_name] = a_file
            print(f"  ✓ Saved: {a_file.name}")
        except Exception as e:
            print(f"  ✗ Error saving SRT: {str(e)}")

        try:
            a_file = output_file_path.with_suffix(".words.srt")
            with open(a_file, "w", encoding="utf-8") as srt:
                write_srt(flatten(result["segments"], "words"), file=srt)
            output_files['.'.join([k.removeprefix('.') for k in a_file.suffixes])][base_name] = a_file
            print(f"  ✓ Saved: {a_file.name}")
        except Exception as e:
            print(f"  ✗ Error saving words SRT: {str(e)}")

    # save TSV
    if "tsv" in output_formats:
        try:
            a_file = output_file_path.with_suffix(".tsv")
            with open(a_file, "w", encoding="utf-8") as csv:
                write_tsv(result["segments"], file=csv)
            output_files['.'.join([k.removeprefix('.') for k in a_file.suffixes])][base_name] = a_file
            print(f"  ✓ Saved: {a_file.name}")
        except Exception as e:
            print(f"  ✗ Error saving TSV: {str(e)}")

        try:
            a_file = output_file_path.with_suffix(".words.tsv")
            with open(a_file, "w", encoding="utf-8") as csv:
                write_tsv(flatten(result["segments"], "words"), file=csv)
            output_files['.'.join([k.removeprefix('.') for k in a_file.suffixes])][base_name] = a_file
            print(f"  ✓ Saved: {a_file.name}")
        except Exception as e:
            print(f"  ✗ Error saving words TSV: {str(e)}")

    return output_files


def process_recordings(recordings_dir: Path, output_dir=None, video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v'], model_path_root: Path = Path(r'F:\AITEMP\whisper_models')):
    # Define the recordings directory
    if isinstance(recordings_dir, str):
        recordings_dir = Path(recordings_dir).resolve()
    print(f'processing_recordings for recordings_dir: "{recordings_dir.as_posix()}"...')
    # Create output directory
    if output_dir is None:
        output_dir = recordings_dir.joinpath('transcriptions').resolve()
        # output_dir = Path("./transcriptions")
    if isinstance(output_dir, str):
        output_dir = Path(output_dir).resolve()

    output_dir.mkdir(exist_ok=True)
    print(f'\t transcriptions will output to output_dir: "{output_dir.as_posix()}"')

    # Get all video files and create alias dir before loading model (so "Found N files" appears quickly)
    video_files = []
    for ext in video_extensions:
        video_files.extend(recordings_dir.glob(f"*{ext}"))
        video_files.extend(recordings_dir.glob(f"*{ext.upper()}"))

    alias_dir = recordings_dir.parent / "edf_video_aliases"
    alias_dir.mkdir(exist_ok=True)

    if not video_files:
        print(f"No video files found in {recordings_dir}")
        return

    print(f"Found {len(video_files)} video files to process")

    # Load the model once (after file discovery so progress is visible sooner)
    model_path_root = model_path_root.resolve()
    assert model_path_root.exists()
    model_name: str = "medium.en"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper model at model_path_root: '{model_path_root.as_posix()}' (device={device})...")
    t0_model = time.perf_counter()
    model = whisper.load_model(model_name, download_root=str(model_path_root), device=device)
    print(f"Whisper model loaded. (Model load: {time.perf_counter() - t0_model:.1f}s)")

    # Preload Silero VAD so first-file transcribe does not stall with no progress
    print("Loading Silero VAD...")
    get_vad_segments(torch.zeros(16000, dtype=torch.float32), method="silero")  # 1s at 16kHz (Whisper SAMPLE_RATE)
    print("Done.")

    output_files = {'json': {}, 'srt': {}, 'csv': {}}
    first_file_timed = True
    failed_files: List[Path] = []
    # Process each video file
    for video_file in video_files:
        print(f"\nProcessing: {video_file.name}")
        base_name = video_file.stem
        try:
            ## try making a symlink with an EDF+ compatible formatted name: https://www.edfplus.info/specs/video.html
            edf_compatible_name = build_EDF_compatible_video_filename(video_file.name)
            print(f'\tedf_compatible_name: "{edf_compatible_name}"')
            edf_compatible_path = alias_dir / edf_compatible_name
            if not edf_compatible_path.exists():
                edf_compatible_path.symlink_to(video_file.resolve())

            found_output_files: List[Path] = find_extant_output_files(output_dir=output_dir, base_name=base_name)
            if found_output_files:
                print(f"  ✗ Skipping {video_file.name} as its outputs already exist: {found_output_files}")
                continue
            if first_file_timed:
                print("  Running VAD and transcription...")
            print("  Loading audio...")
            t0_audio = time.perf_counter()
            audio = whisper.load_audio(str(video_file))
            print("  Audio loaded.")
            if first_file_timed:
                print(f"  First file load_audio: {time.perf_counter() - t0_audio:.1f}s")

            t0_transcribe = time.perf_counter()
            result = whisper.transcribe(model, audio, language="en", vad="silero", remove_empty_words=True)
            if first_file_timed:
                print(f"  First file transcribe: {time.perf_counter() - t0_transcribe:.1f}s")
                first_file_timed = False

            base_name = video_file.stem
            curr_output_files_dict = write_results(result, output_dir=output_dir, base_name=base_name)
            for k, curr_out_files_dict in curr_output_files_dict.items():
                if k not in output_files:
                    output_files[k] = dict()
                output_files[k].update(**curr_out_files_dict)

        except Exception as e:
            failed_files.append(video_file)
            print(f"  ✗ Error processing {video_file.name}: [{type(e).__name__}] {e}")
            continue

    if failed_files:
        print(f"\nProcessing complete with {len(failed_files)} failed file(s): {[f.name for f in failed_files]}")
    print(f"\nProcessing complete! Output files saved to: {output_dir.resolve()}")
    return output_files


if __name__ == "__main__":
    # Format(s) of the output file(s). Possible formats are: txt, vtt, srt, tsv, csv, json. Several formats can be specified by using commas (ex: "json,vtt,srt"). By default ("all"), all available formats  
    recordings_dir = Path(r"M:\ScreenRecordings\EyeTrackerVR_Recordings").resolve() # Debut_%YYYY%-%MM%-%DD%T%HH%%MIN%%SS%
    # video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']
    video_extensions = ['.mp4']
    output_files = process_recordings(recordings_dir=recordings_dir, video_extensions=video_extensions)
    print(f'All processing complete! output_files: {output_files}\n\ndone.')
