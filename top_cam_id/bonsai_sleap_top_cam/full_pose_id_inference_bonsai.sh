#!/bin/bash
#SBATCH --job-name=full_pose_id_inference_bonsai    # job name
#SBATCH --partition=gpu                             # partition (queue)
#SBATCH --gres=gpu:1                                # number of gpus per node
#SBATCH --nodes=1                                   # node count
#SBATCH --exclude=gpu-sr670-20                      # DNN lib missing 
#SBATCH --ntasks=1                                  # total number of tasks across all nodes
#SBATCH --mem=16G                                   # total memory per node 
#SBATCH --time=0-20:00:00                           # total run time limit (DD-HH:MM:SS)
#SBATCH --array=0-1        # array job specification
#SBATCH --output=slurm_output/predict_%N_%j.out     # standard output file path

module load cuda/11.8

USER_EMAIL="a.pouget@ucl.ac.uk"

VIDEO_FILES=("/ceph/aeon/aeon/data/raw/AEON3/social0.2/2024-02-09T16-07-32/CameraTop/CameraTop_2024-02-10T02-00-00.avi")
VIDEO_FILE=${VIDEO_FILES[$SLURM_ARRAY_TASK_ID]}

OUTPUT_DIR=/ceph/aeon/aeon/code/scratchpad/anaya/bonsai-sleap-out/

TIMESTAMP=$(basename "$VIDEO_FILE" | cut -d'_' -f2-3 | cut -d'.' -f1)
OUTPUT_FILE="CameraTop_202_gpu-partition_5539055_2024-08-04T10-15-37_fullpose-id"

echo "Video file: $VIDEO_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Timestamp: $TIMESTAMP"
echo "Output file: $OUTPUT_FILE"

MONO_OUTPUT=$(mono /ceph/aeon/aeon/code/bonsai-sleap/bonsai2.8.2/Bonsai.exe \
    --no-editor \
    /ceph/aeon/aeon/code/bonsai-sleap/example_workflows/combine_sleap_models/combine_sleap_models_aeon.bonsai \
    -p:VideoFile=$VIDEO_FILE \
    -p:OutputDir=$OUTPUT_DIR \
    -p:OutputFile=$OUTPUT_FILE \
    -p:IdModelFile="/ceph/aeon/aeon/code/bonsai-sleap/example_workflows/combine_sleap_models/aeon3_social02_exported_models/id/frozen_graph.pb" \
    -p:IdTrainingConfig="/ceph/aeon/aeon/code/bonsai-sleap/example_workflows/combine_sleap_models/aeon3_social02_exported_models/id/confmap_config.json" \
    -p:PoseModelFile="/ceph/aeon/aeon/code/bonsai-sleap/example_workflows/combine_sleap_models/aeon3_social02_exported_models/full_pose/frozen_graph.pb" \
    -p:PoseTrainingConfig="/ceph/aeon/aeon/code/bonsai-sleap/example_workflows/combine_sleap_models/aeon3_social02_exported_models/full_pose/confmap_config.json" \
    2>&1)
echo "$MONO_OUTPUT"

# Check if MONO_OUTPUT contains any error pattern
if echo "$MONO_OUTPUT" | grep -q "Exception"; then
  echo "Inference failed, sending email to warn user."
  EMAIL_BODY="Inference failed for $VIDEO_FILE.\n\nError details:\n$MONO_OUTPUT"
  echo -e "$EMAIL_BODY" | mail -s "Inference Job Failed" $USER_EMAIL
else
  echo "Inference succeeded."
fi

# Rename the output file (the name attributed by Bonsai has issues) 
find $OUTPUT_DIR -maxdepth 1 \
    -type f \
    -name "*\\${OUTPUT_FILE}_${TIMESTAMP}*" \
    -exec mv {} "${OUTPUT_DIR}/${OUTPUT_FILE}_${TIMESTAMP}.bin" \;

date +%T
