import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import re
import pandas as pd
from pathlib import Path
import subprocess
import cv2
from datetime import timedelta, datetime, time
import json

def find_epochs(
        root: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp
) -> dict:
    epochs = {}
    # Update pattern to match timestamps in AVI filenames
    timestamp_pattern = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})\.avi$")
    root_path = Path(root)
    
    for sub_dir in root_path.iterdir():
        if sub_dir.is_dir():
            # Convert to Path object for consistent handling
            epoch_start_time = pd.to_datetime(sub_dir.name, format="%Y-%m-%dT%H-%M-%S")
            if start_time <= epoch_start_time <= end_time:
                camera_top_path = sub_dir / "CameraTop"
                if camera_top_path.exists():
                    # Find all AVI files
                    avi_files = sorted(
                        f for f in camera_top_path.iterdir()
                        if f.is_file() and f.suffix.lower() == '.avi' and timestamp_pattern.search(f.name)
                    )
                    
                    if avi_files:
                        # Get all timestamps and corresponding AVI files
                        timestamp_to_video = {}
                        for f in avi_files:
                            timestamp_str = timestamp_pattern.search(f.name).group(1)
                            timestamp = pd.to_datetime(timestamp_str, format="%Y-%m-%dT%H-%M-%S")
                            timestamp_to_video[timestamp] = f
                        
                        all_timestamps = sorted(timestamp_to_video.keys())
                        
                        if all_timestamps:
                            # Check if there's only one video in the directory
                            if len(all_timestamps) == 1:
                                # If only one video, use epoch_start_time as the base
                                only_timestamp = all_timestamps[0]
                                only_video_path = timestamp_to_video[only_timestamp]
                                
                                # Calculate video duration
                                only_video = cv2.VideoCapture(str(only_video_path))
                                frame_count = int(only_video.get(cv2.CAP_PROP_FRAME_COUNT))
                                fps = only_video.get(cv2.CAP_PROP_FPS)
                                only_duration = 0
                                if fps > 0:
                                    only_duration = frame_count / fps
                                only_video.release()
                                
                                # Calculate true end time: epoch_start_time + video duration
                                true_end_time = epoch_start_time + timedelta(seconds=only_duration)
                                
                            else:
                                # Get end timestamp from the latest video file
                                end_timestamp = max(all_timestamps)
                                
                                # Get the corresponding end video file
                                end_video_path = timestamp_to_video[end_timestamp]
                                
                                # Calculate end video duration
                                end_duration = 0
                                
                                end_video = cv2.VideoCapture(str(end_video_path))
                                frame_count = int(end_video.get(cv2.CAP_PROP_FRAME_COUNT))
                                fps = end_video.get(cv2.CAP_PROP_FPS)
                                if fps > 0:
                                    end_duration = frame_count / fps
                                end_video.release()
                                
                                # Calculate true end time: end_timestamp + end_duration
                                true_end_time = end_timestamp + timedelta(seconds=end_duration)
                            
                            # Format timestamps as YYYY-MM-DDTHH-MM-SS
                            formatted_start_time = epoch_start_time.strftime("%Y-%m-%dT%H-%M-%S")
                            formatted_end_time = true_end_time.strftime("%Y-%m-%dT%H-%M-%S")
                            
                            epochs[str(sub_dir)] = {
                                'epoch_time': formatted_start_time,  # Change here: Use T-formatted time
                                'start_time': formatted_start_time,
                                'end_time': formatted_end_time
                            }
    
    return epochs

def split_into_24h_periods(epoch_start, epoch_end, boundary_hour=2):
    """
    Split an epoch into 24h periods using a configurable boundary hour
    
    Args:
        epoch_start: Start time as a string in format 'YYYY-MM-DDTHH-MM-SS'
        epoch_end: End time as a string in format 'YYYY-MM-DDTHH-MM-SS'
        boundary_hour: Hour of day to use as the boundary (default: 2 for 2 AM)
        
    Returns:
        List of tuples (start, end) with times in 'YYYY-MM-DDTHH-MM-SS' format
    """
    # Convert strings to datetime objects
    start_dt = pd.to_datetime(epoch_start, format="%Y-%m-%dT%H-%M-%S")
    end_dt = pd.to_datetime(epoch_end, format="%Y-%m-%dT%H-%M-%S")
    
    # Find the first boundary after the start time
    if start_dt.hour < boundary_hour:
        # If start time is before boundary hour, first boundary is same day
        first_boundary = pd.Timestamp(
            year=start_dt.year,
            month=start_dt.month,
            day=start_dt.day,
            hour=boundary_hour,
            minute=0,
            second=0
        )
    else:
        # If start time is after boundary hour, first boundary is next day
        first_boundary = pd.Timestamp(
            year=start_dt.year,
            month=start_dt.month,
            day=start_dt.day,
            hour=boundary_hour,
            minute=0,
            second=0
        ) + pd.Timedelta(days=1)
    
    # List to store all period boundaries
    periods = []
    
    # Add the first period (from start to first boundary)
    if first_boundary <= end_dt:
        periods.append((
            start_dt.strftime("%Y-%m-%dT%H-%M-%S"),
            first_boundary.strftime("%Y-%m-%dT%H-%M-%S")
        ))
        current = first_boundary
    else:
        # If the epoch ends before the first boundary
        periods.append((
            start_dt.strftime("%Y-%m-%dT%H-%M-%S"),
            end_dt.strftime("%Y-%m-%dT%H-%M-%S")
        ))
        return periods
        
    # Add 24-hour periods
    while current + pd.Timedelta(days=1) <= end_dt:
        next_boundary = current + pd.Timedelta(days=1)
        periods.append((
            current.strftime("%Y-%m-%dT%H-%M-%S"),
            next_boundary.strftime("%Y-%m-%dT%H-%M-%S")
        ))
        current = next_boundary
    
    # Add the final period if needed
    if current < end_dt:
        periods.append((
            current.strftime("%Y-%m-%dT%H-%M-%S"),
            end_dt.strftime("%Y-%m-%dT%H-%M-%S")
        ))
    
    return periods


def create_slurm_script(
        root: str,
        start_times: list,
        end_times: list,
        parameters: dict,
        skeleton: dict,
        video_config: dict,
        output_dir: str,
        email: str,
        epoch_times: list = None
) -> str:
    parameters_json = json.dumps(parameters)
    skeleton_json = json.dumps(skeleton)
    video_config_json = json.dumps(video_config)
    
    start_times_str = ' '.join([f'"{start_time}"' for start_time in start_times])
    end_times_str = ' '.join([f'"{end_time}"' for end_time in end_times])
    
    # Format epoch_times if provided
    epoch_times_str = ""
    epoch_check_condition = "false"
    if epoch_times:
        # Properly format the epoch times for bash arrays
        epoch_times_str = ' '.join([f'"{epoch_time}"' for epoch_time in epoch_times])
        epoch_check_condition = "true"
    
    script = f"""#!/bin/bash
#SBATCH --job-name=tube_test_detect             # job name
#SBATCH --partition=gpu_branco                  # partition (queue)
#SBATCH --cpus-per-task=12                      # CPU cores per task
#SBATCH --mem-per-cpu=16G                       # memory per CPU core
#SBATCH --time=00-12:00:00                      # total run time limit (DD-HH:MM:SS)
#SBATCH --array=0-{len(start_times)-1}          # array job
#SBATCH --output=slurm_output/%N_%j.out         # output file path
#SBATCH --exclude=gpu-sr670-22
date +"%Y-%m-%d %H:%M:%S"
USER_EMAIL="{email}"
STARTS=({start_times_str})
START=${{STARTS[$SLURM_ARRAY_TASK_ID]}}
ENDS=({end_times_str})
END=${{ENDS[$SLURM_ARRAY_TASK_ID]}}

# Handle epoch time if provided
EPOCH_CMD=""
if [ {epoch_check_condition} = true ]; then
    EPOCHS=({epoch_times_str})
    EPOCH="${{EPOCHS[$SLURM_ARRAY_TASK_ID]}}"
    EPOCH_CMD="--epoch_time $EPOCH"
    echo "Processing epoch $EPOCH for period with start time: $START and end time: $END"
else
    echo "Processing period with start time: $START and end time: $END"
fi

# Run the Python script with error handling
python -u detect_tube_tests.py --root {root} --start $START --end $END $EPOCH_CMD --parameters '{parameters_json}' --skeleton '{skeleton_json}' --video_config '{video_config_json}' --output_dir {output_dir}
EXIT_STATUS=$?
if [ $EXIT_STATUS -eq 0 ]; then
    echo "Successfully processed"
else
    echo "Error detected (exit code: $EXIT_STATUS), sending email to warn user."
    if [ {epoch_check_condition} = true ]; then
        EMAIL_BODY="SLURM job with ID ${{SLURM_JOB_ID}}, and array ID ${{SLURM_ARRAY_JOB_ID}} Task ${{SLURM_ARRAY_TASK_ID}} encountered an issue for epoch $EPOCH, period with start time: $START and end time: $END.\\n\\nPlease check the job logs."
    else
        EMAIL_BODY="SLURM job with ID ${{SLURM_JOB_ID}}, and array ID ${{SLURM_ARRAY_JOB_ID}} Task ${{SLURM_ARRAY_TASK_ID}} encountered an issue for period with start time: $START and end time: $END.\\n\\nPlease check the job logs."
    fi
    echo -e "$EMAIL_BODY" | mail -s "Inference Job ${{SLURM_JOB_ID}} Failed" $USER_EMAIL
fi
date +"%Y-%m-%d %H:%M:%S"
"""
    return script

def parse_dict(arg_string):
    try:
        return json.loads(arg_string)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError(f"Invalid dictionary format: {arg_string}")

def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--root",
        help="Root directory.",
        required=True,
        type=str
    )
    parser.add_argument(
        "--start",
        help="Start time of the chunk of interest, to be specified in the format YYYY-MM-DDTHH-MM-SS",
        required=True,
        type=str
    )
    parser.add_argument(
        "--end",
        help="End time of the chunk of interest, to be specified in the format YYYY-MM-DDTHH-MM-SS",
        required=True,
        type=str
    )
    parser.add_argument(
        "--parameters",
        help="The parameters of the detection as a JSON string.",
        required=False,
        type=parse_dict,
        default={
            "angle_tolerance": 45,
            "max_distance_start": 50,
            "max_frame_gap": 20,
            "min_tube_test_start_frames": 15,
            "search_window_seconds": 1,
            "min_distance": 30,
            "max_distance_end": 60,
            "movement_threshold": 2,
            "gate_width": 20
        }
    )
    parser.add_argument(
        "--skeleton", 
        help="A mapping of the required nodes (nose, head, centroid, and tail_base) to their corresponding names in your SLEAP project as a JSON string.",
        required=False,
        type=parse_dict,
        default={
            "nose": "nose",
            "head": "head", 
            "centroid": "spine2",
            "tail_base": "spine4"
        }
    )
    parser.add_argument(
        "--video_config",
        help="The configuration for generating videos as a JSON string.",
        required=False,
        type=parse_dict,
        default={
            "gen_vids": False,
            "video_save_path": None,
            "camera": "CameraTop"
        }
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory for the csv of tube test detection times.",
        required=True,
        type=str
    )
    parser.add_argument(
        "--email",
        help="Email address to send error warnings to",
        required=False,
        type=str,
        default="a.pouget@ucl.ac.uk"
    )
    parser.add_argument(
        "--boundary_hour",
        help="Hour (0-23) to use as a daily boundary when parallelizing. Splits processing into 24-hour chunks. If not specified, processes the entire epoch at once.",
        required=False,
        type=int,
        default=None
    )
    args = vars(parser.parse_args())

    root = Path(args["root"])
    raw_root = Path(str(root).replace("ingest", "raw"))
    start = pd.Timestamp(args["start"]).tz_localize(None)
    end = pd.Timestamp(args["end"]).tz_localize(None)
    parameters = args["parameters"]
    skeleton = args["skeleton"]
    video_config = args["video_config"]
    output_dir = Path(args["output_dir"])
    email = args["email"]
    boundary_hour = args["boundary_hour"]

    # Split the path into parts using pathlib
    root_parts = root.parts
    acquisition_computer = root_parts[-2]
    exp_name = root_parts[-1]

    # Get epoch times
    epochs = find_epochs(raw_root, start, end)
    epoch_times = [epoch['epoch_time'] for epoch in epochs.values()]
    print(f"Found epochs: {epoch_times}")
    
    # Split each epoch into 24h periods if boundary_hour is specified
    all_periods = []
    for epoch_dir, times in epochs.items():
        epoch_start = times['start_time']
        epoch_end = times['end_time']
        
        if boundary_hour is None:
            # If boundary_hour is None, don't split - process the entire epoch
            periods = [(epoch_start, epoch_end)]
            print(f"Processing entire epoch {epoch_dir} as a single period: {periods}")
        else:
            # Otherwise, split into 24h periods using the specified boundary hour
            periods = split_into_24h_periods(epoch_start, epoch_end, boundary_hour)
            print(f"Epoch {epoch_dir} split into {len(periods)} periods: {periods}")
        
        all_periods.extend(periods)
    
    # Extract start and end times for all periods
    start_times = [period[0] for period in all_periods]
    end_times = [period[1] for period in all_periods]
    
    print(f"Total number of periods to process: {len(start_times)}")
    
    # Create the slurm script
    script_args = {
        "root": str(root),  # Convert Path back to string for the script
        "start_times": start_times,
        "end_times": end_times,
        "parameters": parameters,
        "skeleton": skeleton,
        "video_config": video_config,
        "output_dir": str(output_dir),  # Convert Path back to string for the script
        "email": email
    }
    if boundary_hour is None:
        script_args["epoch_times"] = epoch_times
    script = create_slurm_script(**script_args)
    script_filename = f"{acquisition_computer}_{exp_name}_tube_test_detection.sh"
    
    # Write the script to a file
    script_path = Path(script_filename)
    script_path.write_text(script)
    
    # Submit the script to SLURM
    subprocess.run(f"sbatch {script_filename}", shell=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(e)