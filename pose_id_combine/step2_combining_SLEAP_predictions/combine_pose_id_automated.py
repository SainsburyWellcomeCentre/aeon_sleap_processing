import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import re
import pandas as pd
from pathlib import Path
import time
import os
import subprocess

def validate_datetime_format(
        value: str
) -> str:
    pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}$"
    if not re.match(pattern, value):
        raise argparse.ArgumentTypeError(f"Invalid datetime format: '{value}'. Expected format: YYYY-MM-DDTHH-MM-SS")
    return value

def find_epochs(
        root: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp
) -> dict:
    epochs = {}
    timestamp_pattern = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})\.bin$")

    root_path = Path(root)
    for sub_dir in root_path.iterdir():
        if sub_dir.is_dir():
            epoch_time = pd.to_datetime(os.path.basename(sub_dir), format="%Y-%m-%dT%H-%M-%S")
            if start_time <= epoch_time <= end_time:
                camera_top_path = sub_dir / "CameraTop"
                if camera_top_path.exists():
                    bin_files = sorted(
                        f for f in camera_top_path.iterdir()
                        if f.is_file() and f.suffix == '.bin' and timestamp_pattern.search(f.name)
                    )
                    if bin_files:
                        times = sorted({
                            timestamp_pattern.search(f.name).group(1)
                            for f in bin_files
                        })
                        epochs[str(sub_dir)] = times
    return epochs

def create_slurm_script(
        root: str,
        chunks_to_process: list,
        fps: int,
        output_dir: str,
        email: str,
        job_id: str = None 
) -> str:
    chunks_to_process_str = ' '.join([f'"{chunk}"' for chunk in chunks_to_process])
    job_id_str = f"{job_id}" if job_id is not None else ""

    script = f"""#!/bin/bash
#SBATCH --job-name=ID_pose_match                # job name
#SBATCH --partition=cpu                         # partition (queue)
#SBATCH --mem=4G                                # total memory per node
#SBATCH --time=00-02:00:00                      # total run time limit (DD-HH:MM:SS)
#SBATCH --array=0-{len(chunks_to_process)-1}    # array job
#SBATCH --output=slurm_output/%N_%j.out         # output file path

date +"%Y-%m-%d %H:%M:%S"
USER_EMAIL="{email}"
CHUNKS=({chunks_to_process_str})
CHUNK=${{CHUNKS[$SLURM_ARRAY_TASK_ID]}}
JOB_ID="{job_id_str}"
if [ -z "$JOB_ID" ]; then
    JOB_ID=$SLURM_ARRAY_JOB_ID
fi

echo "Processing chunk $CHUNK"
# Run the Python script with error handling
if python combine_pose_id_chunk.py --root {root} --start $CHUNK --fps {fps} --output_dir {output_dir} --job_id $JOB_ID; then
    echo "Successfully processed chunk $CHUNK"
else
    echo "Error detected, sending email to warn user."
    EMAIL_BODY="SLURM job ${{SLURM_JOB_ID}} Task ${{SLURM_ARRAY_TASK_ID}} with ID ${{SLURM_JOB_ID}} encountered an issue for ${{CHUNK}}.\\n\\nPlease check the job logs."
    echo -e "$EMAIL_BODY" | mail -s "Inference Job ${{SLURM_JOB_ID}} Failed" $USER_EMAIL
fi
date +"%Y-%m-%d %H:%M:%S"
"""
    return script

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
        "--fps",
        help="Frames per second used for computing end time offset (default: 50).",
        required=False,
        type=float,
        default=50
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory for the processed data",
        required=True,
        type=str
    )
    parser.add_argument(
        "--job_id",
        help="Job ID used for the matched pose ID data config file, to be specified if you want it to match a set of already processed files",
        required=False,
        type=int,
        default=None
    )
    parser.add_argument(
        "--email",
        help="Email address to send error warnings to",
        required=False,
        type=str,
        default="a.pouget@ucl.ac.uk"
    )
    args = vars(parser.parse_args())

    root = args["root"]
    fps = args["fps"]
    start = pd.Timestamp(args["start"]).tz_localize(None)
    end = pd.Timestamp(args["end"]).tz_localize(None)
    output_dir = args["output_dir"]
    job_id = args["job_id"]
    email = args["email"]

    root_split = re.split(r"[\\/]", root) 
    acquisition_computer = root_split[-2]
    exp_name = root_split[-1]

    # Dictionary to track processed chunks for each epoch
    processed_chunks = {}

    while True:
        epochs = find_epochs(root, start, end)
        for epoch, chunks in epochs.items():
            # Initialize the set for this epoch if not already done
            if epoch not in processed_chunks:
                processed_chunks[epoch] = set()
            # Filter out chunks that have already been processed
            new_chunks = [chunk for chunk in chunks if chunk not in processed_chunks[epoch]]
            if not new_chunks:
                print(f"No new chunks to process for epoch {epoch}")
                continue  # Skip if no new chunks to process
            # Mark the new chunks as processed
            processed_chunks[epoch].update(new_chunks)
            print(f"New chunks to process for epoch {epoch}: {new_chunks}")
            # Update output directory for this epoch
            epoch_output_dir = os.path.join(output_dir, acquisition_computer, exp_name, epoch, "CameraTop")
            script_args = {
                "root": root,
                "chunks_to_process": new_chunks,
                "fps": fps,
                "output_dir": epoch_output_dir,
                "email": email,
            }
            if job_id is not None:
                script_args["job_id"] = job_id
            script = create_slurm_script(**script_args)
            script_filename = f"{acquisition_computer}_{exp_name}_{Path(epoch).name}_quad_ID_top_pose_match_automated.sh"
            with open(script_filename, "w") as f:
                f.write(script)
            subprocess.run(f"sbatch {script_filename}", shell=True)
        # Exit or sleep for 30 minutes before checking again 
        if pd.Timestamp.now() > end + pd.Timedelta(hours=6):
            print("Exiting.")
            break
        time.sleep(1800)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(e)
