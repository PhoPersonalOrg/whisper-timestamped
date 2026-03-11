"""
Convert whisper transcript CSV to LSL (Lab Streaming Layer) compatible format
with absolute timestamps and multi-modal analysis support.
"""

import pandas as pd
import numpy as np
import mne
from datetime import datetime, timedelta
from pathlib import Path
import json
from whisper_timestamped.parse_video_filename import parse_video_filename
from typing import List, Dict, Tuple, Optional, Union
# from process_recordings import find_extant_output_files, write_results, process_recordings

class VideoTranscriptToLabStreamingLayer:
    """
    Convert whisper transcript CSV to LSL (Lab Streaming Layer) compatible format
    with absolute timestamps and multi-modal analysis support.
    """

    @classmethod
    def add_absolute_timestamps(cls, segments: List, file_basename: Union[datetime, Path, str]) -> List:
        """
        Parse transcript CSV and add absolute datetime timestamps.

        Args:
            csv_path: Path to the whisper transcript CSV file
            video_filename: Video filename to parse date from. If None, inferred from csv_path.

        Returns:
            DataFrame with absolute datetime columns added
        """
        # We need the recording start absolute datetime, so get this as provided or from the provided basename
        base_datetime = None
        if isinstance(file_basename, datetime):
            ## already have the final form, the datetime
            base_datetime = file_basename
        else:
            if isinstance(file_basename, str):
                file_basename = Path(file_basename)
            file_basename = file_basename.stem  # removes .csv extension

            # Parse the base datetime from video filename
            try:
                base_datetime = parse_video_filename(file_basename)
            except ValueError as e:
                raise ValueError(f"Could not parse datetime from filename '{file_basename}': {e}")

        assert base_datetime is not None
        # Convert relative timestamps to absolute datetimes
        for a_segment in segments:
            a_segment['absolute_start'] = base_datetime + timedelta(seconds=a_segment['start'])
            a_segment['absolute_end'] = base_datetime + timedelta(seconds=a_segment['end']) 

        return segments

    @classmethod
    def create_lsl_stream_data(cls, df: pd.DataFrame, stream_name: str = "transcript", source_id: str = "whisper", stream_save_filename: Optional[Path]=None) -> dict:
        """
        Create LSL-compatible stream data structure from transcript DataFrame.

        Args:
            df: DataFrame with transcript data and absolute timestamps
            stream_name: Name for the LSL stream
            source_id: Source identifier for the stream

        Returns:
            Dictionary containing LSL stream metadata and data
            
            
        Usage:
            lsl_stream_output_path = output_dir / f"{csv_path.stem}.lsl.json"
            lsl_stream_output = VideoTranscriptToLabStreamingLayer.create_lsl_stream_data(file_contents['segments'], stream_save_filename=lsl_stream_output_path)
        """
        start_times = []
        end_times = []
        # Create LSL stream info
        stream_info = {
            "name": stream_name,
            "type": "Markers",
            "channel_count": 1,
            "nominal_srate": 0,  # Irregular sampling
            "channel_format": "string",
            "source_id": source_id,
            "created_at": datetime.now().isoformat(),
            "session_id": f"{source_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }

        ## BEGIN BODY
        # Extract messages and timestamps
        messages = []
        timestamps = []
        samples = []
        if isinstance(df, pd.DataFrame):
            for _, row in df.iterrows():
                # Each sample contains the text and timing info
                sample = {
                    "timestamp": row['absolute_start'].timestamp(),
                    "data": {
                        "text": row['text'],
                        "duration": row['end'] - row['start'],
                        "start_offset": row['start'],
                        "end_offset": row['end'],
                        "confidence": getattr(row, 'confidence', None)  # if available
                    }
                }
                samples.append(sample)
                start_times.append(df['absolute_start'])
                end_times.append(df['absolute_end'])
                # MNE Version:
                message = row['text'] if row['text'] else ''
                timestamp = row['absolute_start'].timestamp() # row['start']
                messages.append(message)
                timestamps.append(timestamp)

        elif isinstance(df, list):
            ## a list of sample dicts
            for row in df:
                # Each sample contains the text and timing info
                sample = {
                    "timestamp": row['absolute_start'].timestamp(),
                    "data": {
                        "text": row['text'],
                        "duration": row['end'] - row['start'],
                        "start_offset": row['start'],
                        "end_offset": row['end'],
                        "confidence": getattr(row, 'confidence', None)  # if available
                    }
                }
                samples.append(sample)
                start_times.append(row['absolute_start'])
                end_times.append(row['absolute_end'])
                # MNE Version:
                message = row['text'] if row['text'] else ''
                timestamp = row['absolute_start'].timestamp() # row['start']
                messages.append(message)
                timestamps.append(timestamp)



        else:
            raise TypeError(f'unexpected type: {type(df)}')

        lsl_data = {
            "stream_info": stream_info,
            "samples": samples,
            "total_samples": len(samples),
            "start_time": np.min(start_times).isoformat(),
            "end_time": np.max(start_times).isoformat()
            # "start_time": df['absolute_start'].min().isoformat(),
            # "end_time": df['absolute_end'].max().isoformat()
        }

        # ==================================================================================================================================================================================================================================================================================== #
        # MNE VERSION                                                                                                                                                                                                                                                                          #
        # ==================================================================================================================================================================================================================================================================================== #
        ## INPUTS: messages, timestamps, messages
        # Convert timestamps to relative times (from first sample)
        if timestamps:
            first_timestamp = timestamps[0]
            relative_timestamps = [ts - first_timestamp for ts in timestamps]
        else:
            relative_timestamps = []
        # Create annotations (MNE's way of handling markers/events)
        # Set orig_time=None to avoid timing conflicts
        annotations = mne.Annotations(
            onset=relative_timestamps,
            duration=[0.0] * len(relative_timestamps),  # Instantaneous events
            description=messages,
            orig_time=None  # This fixes the timing conflict
        )

        # Create a minimal info structure for the markers
        info = mne.create_info(
            ch_names=[f'{stream_name}_Markers'],
            sfreq=1000,  # Dummy sampling rate for the minimal channel
            ch_types=['misc']
        )

        # Create raw object with minimal dummy data
        # We need at least some data points to create a valid Raw object
        if len(timestamps) > 0:
            # Create dummy data spanning the recording duration
            duration = relative_timestamps[-1] if relative_timestamps else 1.0
            n_samples = int(duration * 1000) + 1000  # Add buffer
            dummy_data = np.zeros((1, n_samples))
        else:
            dummy_data = np.zeros((1, 1000))  # Minimum 1 second of data

        raw = mne.io.RawArray(dummy_data, info)

        # Set measurement date to match the first timestamp
        if timestamps:
            raw.set_meas_date(timestamps[0])

        raw.set_annotations(annotations)

        # Add metadata to the raw object
        raw.info['description'] = 'VideoTranscription LSL Stream Recording'
        raw.info['experimenter'] = 'PhoVideoTranscriptToLabStreamingLayer'


        if stream_save_filename is not None:
            xdf_filename = stream_save_filename
            if isinstance(xdf_filename, Path):
                xdf_filename = xdf_filename.as_posix()

            # Determine output filename and format
            if xdf_filename.endswith('.xdf'):
                # Save as FIF (MNE's native format)
                fif_filename = xdf_filename.replace('.xdf', '.fif')
                raw.save(fif_filename, overwrite=True)
                actual_filename = fif_filename
                file_type = "FIF"
            else:
                # Use the original filename
                raw.save(xdf_filename, overwrite=True)
                actual_filename = xdf_filename
                file_type = "FIF"

            # Create LSL stream
            # cls.save_lsl_stream(lsl_data, output_path=stream_save_filename)
            
        return lsl_data, raw
    

    # @classmethod
    # def save_lsl_stream(cls, stream_data: dict, output_path: Union[str, Path]) -> None:
    #     """Save LSL stream data to JSON file."""
    #     output_path = Path(output_path)
    #     with open(output_path, 'w', encoding='utf-8') as f:
    #         json.dump(stream_data, f, indent=2, ensure_ascii=False)
                

    @classmethod
    def MAIN_process_all_transcripts(cls, recordings_dir: Path, output_extensions = ['.json']):
        """ Main function - parses all exported transcripts in a directory and produces new LabStreamingLayer (LSL) XDF/FIF streams that are saved to the 'LSL_Converted' subdirectory (which is created if needed)

        lsl_stream_output_path, found_valid_output_files, read_valid_output_files_dict = VideoTranscriptToLabStreamingLayer.MAIN_process_all_transcripts(recordings_dir = Path(r"M:\ScreenRecordings\EyeTrackerVR_Recordings").resolve())

        """
        # from whisper.utils import read_csv

        # recordings_dir = Path(r"M:\ScreenRecordings\EyeTrackerVR_Recordings").resolve()

        # Define the recordings directory
        if isinstance(recordings_dir, str):
            recordings_dir = Path(recordings_dir).resolve()
        print(f'processing_recordings for recordings_dir: "{recordings_dir.as_posix()}"...')
        # Create output directory
        output_dir: Path = recordings_dir.joinpath('transcriptions').resolve()
        lsl_converted_streams_output_dir: Path = output_dir.joinpath('LSL_Converted')
        lsl_converted_streams_output_dir.mkdir(exist_ok=True)

        # output_dir = Path("./transcriptions")

        # output_extensions = ['.txt', '.vtt', '.srt', '.tsv', '.csv', '.json']
        
        # output_extensions = ['.csv']

        found_output_files: List[Path] = []
        for ext in output_extensions:
            found_output_files.extend(output_dir.glob(f"*{ext}"))


        # found_output_files: List[Path] = find_extant_output_files(output_dir=output_dir, base_name=base_name, output_formats=output_formats)
        found_output_files

        found_valid_output_files = []
        read_valid_output_files_dict = {}
        output_lsl_fif_files = []

        for a_file in found_output_files:
            if a_file.exists() and a_file.is_file():
                # file_contents = json.load(a_file)
                with open(a_file, "r", encoding="utf-8") as js:
                    file_contents = json.load(js) # (result, js, indent=2, ensure_ascii=False)
                if file_contents:
                    is_ready_for_LSL_stream_export: bool = False
                    if len(file_contents['segments']) > 0:
                        try:
                            file_contents['segments'] = cls.add_absolute_timestamps(segments=file_contents['segments'], file_basename=a_file.stem)
                            print(f"\nSuccess! '{a_file.as_posix()}'\n\tProcessed {len(file_contents['segments'])} transcript segments")
                            is_ready_for_LSL_stream_export = True
                        except Exception as e:
                            print(f"Failed to parse to LabStreamingLayer for file: '{a_file.as_posix()}' Error: {e}")
                            is_ready_for_LSL_stream_export = False
                            pass

                        if is_ready_for_LSL_stream_export:            
                            try:
                                # lsl_stream_output_path = lsl_converted_streams_output_dir / f"{a_file.stem}.lsl.json"
                                lsl_stream_output_path = lsl_converted_streams_output_dir / f"{a_file.stem}.lsl.fif"
                                lsl_stream_output, raw_lsl_stream_output = cls.create_lsl_stream_data(file_contents['segments'], stream_save_filename=lsl_stream_output_path)                
                                print(f"\tSuccess exporting to LSL Stream! '{lsl_stream_output_path.as_posix()}'")
                                output_lsl_fif_files.append(lsl_stream_output_path)
                            except Exception as e:
                                print(f"\tFailed to export final LabStreamingLayer stream to '{lsl_stream_output_path.as_posix()}' for source file: '{a_file.as_posix()}' Error: {e}")
                                raise

                    # print(f'file_contents: {file_contents}')    

                # _a_read_text: str = a_file.read_text()
                # if _a_read_text:

                #     found_valid_output_files.append(a_file)
                #     read_valid_output_files_dict[a_file.name] = _a_read_text
                #     ## actually include the file
                #     try:
                #         df, lsl_data = process_transcript_to_lsl(csv_path=a_file, output_dir=output_dir, video_filename=a_file.name)
                #         print(f"\nSuccess! Processed {len(df)} transcript segments")
                #     except Exception as e:
                #         print(f"Failed to parse to LabStreamingLayer for file: '{a_file.as_posix()}' Error: {e}")
                #         pass

        return output_lsl_fif_files, found_valid_output_files, read_valid_output_files_dict




# Example usage and CLI interface
if __name__ == "__main__":
    import sys
    
    # if len(sys.argv) < 2:
    #     print("Usage: python transcript_to_lsl.py <recordings_path>")
    #     sys.exit(1)
    
    recordings_dir = sys.argv[1] if len(sys.argv) > 1 else None
    # output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    # video_filename = sys.argv[3] if len(sys.argv) > 3 else None
    if recordings_dir is None:
        ## try default    
        recordings_dir = Path(r"M:\ScreenRecordings\EyeTrackerVR_Recordings").resolve()
        
    try:
        # Define the recordings directory
        if isinstance(recordings_dir, str):
            recordings_dir = Path(recordings_dir).resolve()
            
        assert recordings_dir.exists()
        assert recordings_dir.is_dir()
        print(f'processing_recordings for recordings_dir: "{recordings_dir.as_posix()}"...')
        # Create output directory
        output_dir: Path = recordings_dir.joinpath('transcriptions').resolve()
        lsl_converted_streams_output_dir: Path = output_dir.joinpath('LSL_Converted')
        lsl_converted_streams_output_dir.mkdir(exist_ok=True)

        print(f'all lsl converted transcripts will be saved to lsl_converted_streams_output_dir: "{lsl_converted_streams_output_dir.as_posix()}"...')
        output_lsl_fif_files, found_valid_output_files, read_valid_output_files_dict = VideoTranscriptToLabStreamingLayer.MAIN_process_all_transcripts(recordings_dir=recordings_dir)
        print(f'\nSuccess! Processed {len(output_lsl_fif_files)} transcript segments\n Exported to lsl_converted_streams_output_dir: "{lsl_converted_streams_output_dir.as_posix()}"\n')
        print("LSL stream data ready for multi-modal analysis")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
