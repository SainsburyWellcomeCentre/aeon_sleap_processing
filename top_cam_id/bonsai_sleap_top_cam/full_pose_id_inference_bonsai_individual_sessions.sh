#!/bin/bash
#SBATCH --job-name=full_pose_id_inference_bonsai    # job name
#SBATCH --partition=a100                            # partition (queue)
#SBATCH --gres=gpu:1                                # number of gpus per node
#SBATCH --exclude=gpu-sr670-20                      # DNN lib missing
#SBATCH --nodes=1                                   # node count
#SBATCH --ntasks=1                                  # total number of tasks across all nodes
#SBATCH --mem=16G                                   # total memory per node 
#SBATCH --time=0-20:00:00                           # total run time limit (DD-HH:MM:SS)
#SBATCH --array=0-68        # array job specification
#SBATCH --output=slurm_output/predict_%N_%j.out     # standard output file path
#SBATCH --mail-type=TIME_LIMIT                      # Send email on job failure
#SBATCH --mail-user=a.pouget@ucl.ac.uk                         # Email address to send notifications

mkdir -p slurm_output
SLURM_OUTPUT="slurm_output/predict_${SLURMD_NODENAME}_${SLURM_JOB_ID}.out"
echo "SLURM output file: $SLURM_OUTPUT"

USER_EMAIL="a.pouget@ucl.ac.uk"

# Load the module and check for failure
if ! module load cuda/11.8; then
  echo "Module load failed on node $SLURMD_NODENAME, sending email to warn user."
  EMAIL_SUBJECT="Module Load Failed on $SLURMD_NODENAME"
  EMAIL_BODY="Module load of cuda/11.8 failed on node $SLURMD_NODENAME."
  echo "$EMAIL_BODY" | mail -s "$EMAIL_SUBJECT" $USER_EMAIL
fi

VIDEO_FILES=("/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-16T19-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-17T07-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-18T19-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-18T04-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-19T10-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-16T14-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-18T23-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-17T23-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-17T05-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-19T00-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-18T06-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-17T20-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-17T18-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-17T00-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-18T18-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-17T09-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-18T09-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-17T06-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-19T08-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-18T16-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-17T10-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-17T01-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-18T07-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-17T16-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-18T17-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-16T20-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-16T15-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-17T15-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-18T20-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-17T17-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-18T03-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-17T14-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-18T00-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-16T18-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-19T09-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-18T12-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-18T10-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-17T22-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-16T23-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-18T22-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-17T08-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-19T02-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-16T21-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-17T04-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-18T01-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-19T06-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-18T15-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-17T12-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-19T04-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-18T05-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-18T13-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-19T03-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-17T21-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-19T07-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-17T11-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-17T19-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-18T08-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-19T01-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-17T13-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-17T02-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-16T22-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-16T17-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-18T11-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-16T16-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-19T05-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-17T03-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-18T14-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-18T21-00-00.avi" "/ceph/aeon/aeon/data/raw/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/CameraTop_2024-08-18T02-00-00.avi")
VIDEO_FILE=${VIDEO_FILES[$SLURM_ARRAY_TASK_ID]}

OUTPUT_DIR=/ceph/aeon/aeon/data/processed/AEON4/social0.4/2024-08-16T14-30-28/CameraTop/

TIMESTAMP=$(basename "$VIDEO_FILE" | cut -d'_' -f2-3 | cut -d'.' -f1)
OUTPUT_FILE="CameraTop_202_gpu-partition_5672002_2024-10-16T10-25-15_fullpose-id"

echo "Video file: $VIDEO_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Timestamp: $TIMESTAMP"
echo "Output file: $OUTPUT_FILE"

date +%T

mono /ceph/aeon/aeon/code/bonsai-sleap/bonsai2.8.2/Bonsai.exe \
    --no-editor \
    /ceph/aeon/aeon/code/bonsai-sleap/example_workflows/combine_sleap_models/bonsai_sleap_individual.bonsai \
    -p:VideoFile=$VIDEO_FILE \
    -p:OutputDir=$OUTPUT_DIR \
    -p:OutputFile=$OUTPUT_FILE \
    -p:Id="BAA-1104797" \
    -p:PoseModelFile="/ceph/aeon/aeon/code/bonsai-sleap/example_workflows/combine_sleap_models/aeon4_social04_exported_models/full_pose_single_subject/frozen_graph.pb" \
    -p:PoseTrainingConfig="/ceph/aeon/aeon/code/bonsai-sleap/example_workflows/combine_sleap_models/aeon4_social04_exported_models/full_pose_single_subject/confmap_config.json" \

# Check for errors in the output
if grep -q "Exception" "$SLURM_OUTPUT"; then
  echo "Inference failed, sending email to warn user."
  EMAIL_SUBJECT="Inference Job Failed on $SLURMD_NODENAME"
  EMAIL_BODY="Inference failed for $VIDEO_FILE.\n\nError details:\n$(cat "$SLURM_OUTPUT")"
  echo -e "$EMAIL_BODY" | mail -s "Inference Job Failed" $USER_EMAIL
else
  echo "Inference succeeded."
  # Rename the output file (the name attributed by Bonsai has issues) 
  find $OUTPUT_DIR -maxdepth 1 \
    -type f \
    -name "*\\${OUTPUT_FILE}_${TIMESTAMP}*" \
    -exec mv {} "${OUTPUT_DIR}/${OUTPUT_FILE}_${TIMESTAMP}.bin" \;
fi

date +%T
