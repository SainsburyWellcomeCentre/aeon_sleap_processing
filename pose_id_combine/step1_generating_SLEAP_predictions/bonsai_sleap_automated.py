import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd
import subprocess
from pathlib import Path
import re
import os
import json
import shutil
from datetime import datetime
import time
import logging
from typing import Optional

def validate_datetime_format(
        value: str
) -> str:
    pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}$"
    if not re.match(pattern, value):
        raise argparse.ArgumentTypeError(f"Invalid datetime format: '{value}'. Expected format: YYYY-MM-DDTHH-MM-SS")
    return value

def get_latest_job_id() -> int:
    try:
        # Obtain the latest job ID from the SLURM database
        result = subprocess.check_output("sacct -n -X --format=JobID --allusers | grep -Eo '^[0-9]+' | tail -n 1", shell=True)
        # Decode the output and strip any extra whitespace
        job_id = result.decode('utf-8').strip()
        # If job id cannot be converted to int, return None
        try:
            job_id = int(job_id)
        except ValueError:
            print(f"Failed to convert job ID to integer: {job_id}")
            return None
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while fetching the latest job ID: {e}")
        return None

def find_epochs(
        root: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp
) -> list:
    epochs = []
    root_path = Path(root)
    for sub_dir in root_path.iterdir():
        if sub_dir.is_dir():
            epoch_path = str(sub_dir)
            epoch_time = pd.to_datetime(os.path.basename(sub_dir), format="%Y-%m-%dT%H-%M-%S")
            if start_time <= epoch_time <= end_time:
                epochs.append(epoch_path)
    return epochs

def find_chunks(
        root: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp
) -> dict:
    cameras = ["CameraNorth", "CameraSouth", "CameraEast", "CameraWest", "CameraTop"]
    chunks = {camera: [] for camera in cameras}  # Initialize empty lists for each camera
    timestamp_pattern = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})\.avi$")
    for camera in cameras:
        camera_path = Path(root) / camera
        if not camera_path.exists():
            continue
        avi_files = [
            str(f) for f in sorted(camera_path.iterdir())
            if f.is_file() and f.suffix == '.avi' and (
                (match := timestamp_pattern.search(f.name)) and
                start_time <= pd.to_datetime(match.group(1), format="%Y-%m-%dT%H-%M-%S") <= end_time
            )
        ]
        chunks[camera].extend(avi_files)
    return chunks

def create_slurm_script(
        workflow: str,
        partition: str,
        chunks_to_process: list,
        email: str, 
        output_file_prefix: str,
        camera: str,
        acquisition_computer: str,
        exp_name: str
) -> str:
    chunks_to_process_str = ' '.join(f'"{chunk}"' for chunk in chunks_to_process)
    if workflow == "InferPoses_single_basic.bonsai":
        model_file_name = f"/ceph/aeon/aeon/code/bonsai-sleap/example_workflows/match_quad_id_to_top_cam_pose_parse/top_cam_full_pose_exported_model/frozen_graph.pb"
        training_config = f"/ceph/aeon/aeon/code/bonsai-sleap/example_workflows/match_quad_id_to_top_cam_pose_parse/top_cam_full_pose_exported_model/confmap_config.json"
    else:
        model_file_name = f"/ceph/aeon/aeon/code/bonsai-sleap/example_workflows/match_quad_id_to_top_cam_pose_parse/{camera}_cam_id_exported_models/{acquisition_computer.lower()}_{exp_name.replace('.', '')}/frozen_graph.pb"
        training_config = f"/ceph/aeon/aeon/code/bonsai-sleap/example_workflows/match_quad_id_to_top_cam_pose_parse/{camera}_cam_id_exported_models/{acquisition_computer.lower()}_{exp_name.replace('.', '')}/confmap_config.json"
    script = f"""#!/bin/bash
#SBATCH --job-name=bonsai_sleap                 # job name
#SBATCH --partition={partition}                 # partition (queue)
#SBATCH --gres=gpu:1                            # number of gpus per node
#SBATCH --nodes=1                               # node count
#SBATCH --exclude=gpu-sr670-20                  # fail to query GPU memory from nvidia-smi
#SBATCH --ntasks=1                              # total number of tasks across all nodes
#SBATCH --mem=20G                               # total memory per node
#SBATCH --time=10-00:00:00                      # total run time limit (DD-HH:MM:SS)
#SBATCH --array=0-{len(chunks_to_process)-1}    # array job
#SBATCH --output=slurm_output/%N_%j.out         # output file path

date +"%Y-%m-%d %H:%M:%S"

module load cuda/11.8

nvidia-smi

export LD_LIBRARY_PATH=/ceph/apps/ubuntu-20/packages/cuda/11.8.0_520.61.05/lib64:$LD_LIBRARY_PATH

USER_EMAIL="{email}"

CHUNKS=({chunks_to_process_str})
CHUNK=${{CHUNKS[$SLURM_ARRAY_TASK_ID]}}
TIMESTAMP=$(basename "$CHUNK" | cut -d'_' -f2-3 | cut -d'.' -f1)
OUTPUT_FILE_NAME="{output_file_prefix}_${{TIMESTAMP}}.bin.tmp"

echo "Chunk to process: $CHUNK"
echo "Output directory: $OUTPUT_FILE_NAME"

# sstat -j ${{SLURM_JOB_ID}}.batch --format=JobID,MaxRSS
(
  while true; do
    echo "=== Memory usage at $(date +"%Y-%m-%d %H:%M:%S") ==="

    # Print PID, RSS (in GB), and command for *all* processes owned by $USER
    # Sort by RSS descending and convert KB -> GB in AWK
    ps -U "$USER" -o pid,rss,cmd --sort=-rss \
        | awk '
        BEGIN {{
            sum_rss_gb = 0
        }}

        NR == 1 {{
            # Print a header for clarity.
            print "PID\\tRSS (GB)\\tCOMMAND"
            next
        }}

        {{
            pid = $1
            rss_kb = $2
            # Rebuild the command line (fields 3 onward).
            $1=""  # remove pid
            $2=""  # remove rss
            cmd = $0
            sub(/^[ \\t]+/, "", cmd)  # strip leading whitespace if any

            rss_gb = rss_kb / (1024 * 1024)

            # Only display/accumulate if RSS > 1 GB
            if (rss_gb > 1) {{
                printf "%s\\t%.2f\\t%s\\n", pid, rss_gb, cmd
                sum_rss_gb += rss_gb
            }}
        }}

        END {{
            # Print the accumulated total usage (only processes over 1 GB)
            printf "\\nTotal memory usage (GB) for processes > 1GB: %.2f\\n", sum_rss_gb
        }}
        '
    echo

    sstat -j "${{SLURM_JOB_ID}}.batch" --format=JobID,MaxRSS --noheader | awk '
    {{
    jobid = $1
    val   = $2      # e.g., "12345K", "512M", "3.2G", or even just "12345" (KB assumed)
    
    # Skip lines without a numeric field
    if (val ~ /[0-9]/) {{
        # Last character might be K/M/G. Extract it:
        suffix = substr(val, length(val), 1)
        # Everything except the last character is the numeric portion:
        numeric_val_str = substr(val, 1, length(val)-1)
        
        # If the last char is not K/M/G, just assume "val" is pure kilobytes
        if (suffix !~ /[KMG]/) {{
        suffix = "K"
        numeric_val_str = val
        }}
        
        # Convert numeric portion to float
        numeric_val = numeric_val_str + 0
        
        # Convert to GB
        if (suffix == "K") {{
        maxrss_gb = numeric_val / (1024*1024)
        }} else if (suffix == "M") {{
        maxrss_gb = numeric_val / 1024
        }} else if (suffix == "G"){{
        maxrss_gb = numeric_val
        }} else {{
        # Fallback—treat as KB
        maxrss_gb = numeric_val / (1024*1024)
        }}
        
        # Print the result
        printf "Slurm Job Step: %s  MaxRSS: %.2f GB\\n", jobid, maxrss_gb
    }}
    }}
    '
    echo
    
    sleep 60
  done
) &
BG_MEM_PID=$!

mono /ceph/aeon/aeon/code/bonsai-sleap/bonsai_quad_cam_full_pose_id/Bonsai.exe \\
    --no-editor bonsai_workflows/{workflow} \\
    -p:FileName=$CHUNK \\
    -p:ModelFileName={model_file_name} \\
    -p:OutputFileString=$OUTPUT_FILE_NAME \\
    -p:WriterFileName=$OUTPUT_FILE_NAME \\
    -p:TrainingConfig={training_config}
kill $BG_MEM_PID

# Check for errors in the output log file
# Ignore if bonsai aborts because the .bin file already exists
SLURM_OUT_FILE="slurm_output/${{SLURM_JOB_NODELIST}}_${{SLURM_JOB_ID}}.out"
if grep -q "Exception" "$SLURM_OUT_FILE" && ! grep -q "ABORTING!" "$SLURM_OUT_FILE"; then
  echo "Error detected, sending email to warn user."
  EMAIL_BODY="SLURM job ${{SLURM_JOB_ID}} encountered an issue for $CHUNK.\n\nPlease check the job logs."
  echo -e "$EMAIL_BODY" | mail -s "Inference Job ${{SLURM_JOB_ID}} Failed" $USER_EMAIL
fi

# Rename the output file to remove the .tmp extension
# Or create an empty bin file if it doesn't exist (no poses but good to record that the inference was run)
TARGET_FILE="${{OUTPUT_FILE_NAME%.tmp}}"
if [ -f "$OUTPUT_FILE_NAME" ]; then
    # The .tmp file exists, so rename (remove .tmp extension).
    mv "$OUTPUT_FILE_NAME" "$TARGET_FILE"
    echo "Renamed '$OUTPUT_FILE_NAME' to '$TARGET_FILE'"
elif [ ! -f "$TARGET_FILE" ]; then
    # Neither the .tmp file nor the no-extension file exists, so create empty.
    touch "$TARGET_FILE"
    echo "No .tmp file found; created an empty file: '$TARGET_FILE'"
else
    # The .tmp file doesn't exist, but the target file already exists.
    echo "Nothing to do: '$TARGET_FILE' already exists"
fi

date +"%Y-%m-%d %H:%M:%S"
"""
    return script

def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--root",
        help="Root directory for the raw experiment data",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--start",
        help="Start time of the social period, to be specified in the format YYYY-MM-DDTHH-MM-SS",
        required=True,
        type=str
    )
    parser.add_argument(
        "--end",
        help="End time of the social period, to be specified in the format YYYY-MM-DDTHH-MM-SS",
        required=True,
        type=str
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory for the processed data",
        required=True,
        type=str
    )
    parser.add_argument(
        "--partition",
        help="Partition to run the job on",
        required=False,
        type=str,
        default="a100"
    )
    parser.add_argument(
        "--job_id",
        help="Quad camera ID data job ID, to be specified if you want it to match a set of already processed files",
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
    output_dir = args["output_dir"]
    root_split = re.split(r'[\\/]', root) # Split on back and forward slashes
    acquisition_computer = root_split[-2]
    exp_name =root_split[-1] 
    print(exp_name, acquisition_computer)
    
    start = pd.Timestamp(args["start"])
    end = pd.Timestamp(args["end"])
    partition = args["partition"]
    if args["job_id"] is not None:
        quad_cam_id_job_id = args["job_id"]
    else:
        # Generate fake job ID
        quad_cam_id_job_id = get_latest_job_id()
        if quad_cam_id_job_id is None:
            raise RuntimeError("Exiting.")
    email = args["email"]

    # Create full pose ID config file 
    quad_cam_id_config_dir = f"/ceph/aeon/aeon/data/processed/202/{quad_cam_id_job_id}"
    # Avoids duplicate job IDs if you run the script multiple times in close succession
    while os.path.exists(quad_cam_id_config_dir) and args["job_id"] is None:
        quad_cam_id_job_id += 1
        quad_cam_id_config_dir = f"/ceph/aeon/aeon/data/processed/202/{quad_cam_id_job_id}"
    top_cam_id_job_id = quad_cam_id_job_id + 1
    top_cam_id_config_dir = f"/ceph/aeon/aeon/data/processed/202/{top_cam_id_job_id}"
    top_cam_pose_job_id = quad_cam_id_job_id + 2
    top_cam_pose_config_dir = f"/ceph/aeon/aeon/data/processed/212/{top_cam_pose_job_id}"
    if not os.path.exists(quad_cam_id_config_dir):
        print(f"Copying quad camera ID config file to {quad_cam_id_config_dir}")
        os.makedirs(quad_cam_id_config_dir)
        original_quad_cam_id_config_dir = f"/ceph/aeon/aeon/code/bonsai-sleap/example_workflows/match_quad_id_to_top_cam_pose_parse/quad_cam_id_exported_models/{acquisition_computer.lower()}_{exp_name.replace('.', '')}/confmap_config.json"
        shutil.copy(original_quad_cam_id_config_dir, quad_cam_id_config_dir)
    else:  
        print(f"Quad camera ID config file already exists: {quad_cam_id_config_dir}")
    if not os.path.exists(top_cam_id_config_dir):
        print(f"Copying top camera ID config file to {top_cam_id_config_dir}")
        os.makedirs(top_cam_id_config_dir)
        original_top_cam_id_config_dir = f"/ceph/aeon/aeon/code/bonsai-sleap/example_workflows/match_quad_id_to_top_cam_pose_parse/top_cam_id_exported_models/{acquisition_computer.lower()}_{exp_name.replace('.', '')}/confmap_config.json"
        shutil.copy(original_top_cam_id_config_dir, top_cam_id_config_dir)
    else:
        print(f"Top camera ID config file already exists: {top_cam_id_config_dir}")
    if not os.path.exists(top_cam_pose_config_dir):
        print(f"Copying pose config file to {top_cam_pose_config_dir}")
        os.makedirs(top_cam_pose_config_dir)
        original_top_cam_pose_config_dir = f"/ceph/aeon/aeon/code/bonsai-sleap/example_workflows/match_quad_id_to_top_cam_pose_parse/top_cam_full_pose_exported_model/confmap_config.json"
        shutil.copy(original_top_cam_pose_config_dir, top_cam_pose_config_dir)
    else:
        print(f"Pose config file already exists: {top_cam_pose_config_dir}")

    processed_chunks = {}  # Dictionary to track processed chunks per epoch and camera

    while True:
        epochs_dirs = find_epochs(root=root, start_time=start, end_time=end)
        for epoch_dir in epochs_dirs:
            epoch = Path(epoch_dir).name
            if epoch not in processed_chunks:
                processed_chunks[epoch] = {}  # initialize for each epoch

            # Get list of all chunks to process for this epoch
            chunk_dirs = find_chunks(root=epoch_dir, start_time=start, end_time=end)
            print(f"Processing epoch: {epoch_dir}")
            for camera_key, chunks in chunk_dirs.items():
                if camera_key not in processed_chunks[epoch]:
                    processed_chunks[epoch][camera_key] = set()  # initialize set for each camera
                # Filter out chunks already processed
                new_chunks = [chunk for chunk in chunks if chunk not in processed_chunks[epoch][camera_key]]
                if not new_chunks:
                    print(f"No new chunks to process for epoch {epoch} and camera {camera_key}")
                    continue  # Skip if no new chunks for this camera
                # Mark these chunks as processed
                processed_chunks[epoch][camera_key].update(new_chunks)
                print(f"New chunks to process for epoch {epoch} and camera {camera_key}: {new_chunks}")
                output_dir_camera = os.path.join(output_dir, acquisition_computer, exp_name, epoch, camera_key)
                # Run ID inference
                if camera_key == "CameraTop":
                    camera = "top"
                    job_id_used = top_cam_id_job_id
                else:
                    camera = "quad"
                    job_id_used = quad_cam_id_job_id
                script = create_slurm_script(
                    workflow="InferIdentities_single_basic.bonsai",
                    partition=partition,
                    chunks_to_process=new_chunks,
                    email=email,
                    output_file_prefix=os.path.join(output_dir_camera, f"{camera_key}_202_{job_id_used}"),
                    camera=camera,
                    acquisition_computer=acquisition_computer,
                    exp_name=exp_name
                )
                script_filename = f"{acquisition_computer}_{exp_name}_{camera_key}_ID_bonsai_sleap_inference.sh"
                with open(script_filename, "w") as f:
                    f.write(script)
                subprocess.run(f"sbatch {script_filename}", shell=True)
                if camera_key == "CameraTop":
                    # Run pose inference for CameraTop
                    job_id_used = top_cam_pose_job_id
                    script = create_slurm_script(
                        workflow="InferPoses_single_basic.bonsai",
                        partition=partition,
                        chunks_to_process=new_chunks,
                        email=email,
                        output_file_prefix=os.path.join(output_dir_camera, f"{camera_key}_212_{job_id_used}"),
                        camera=camera,
                        acquisition_computer=acquisition_computer,
                        exp_name=exp_name
                    )
                    script_filename = f"{acquisition_computer}_{exp_name}_CameraTop_pose_bonsai_sleap_inference.sh"
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