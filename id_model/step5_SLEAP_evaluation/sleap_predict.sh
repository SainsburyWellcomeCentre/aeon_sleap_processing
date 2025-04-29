#!/bin/bash
#SBATCH --job-name=sleap_predict        # job name
#SBATCH --partition=gpu                 # partition (queue)
#SBATCH --gres=gpu:1                    # number of gpus per node
#SBATCH --nodes=1                       # node count 
#SBATCH --exclude=gpu-sr670-20          # fail to query GPU memory from nvidia-smi
#SBATCH --ntasks=1                      # total number of tasks across all nodes
#SBATCH --mem=32G                       # total memory per node 
#SBATCH --time=2-00:00:00               # total run time limit (DD-HH:MM:SS)
#SBATCH --output=slurm_output/%N_%j.out # output file path

mkdir -p slurm_output

python -c "import sleap; sleap.versions(); sleap.system_summary()"

sleap-track --model "/ceph/aeon/aeon/code/scratchpad/sleap/multi_point_tracking/multi_animal_CameraNSEW/models/aeon3_social02_labelled_topdown_top.centroid" \
    --model "/ceph/aeon/aeon/code/scratchpad/sleap/multi_point_tracking/multi_animal_CameraNSEW/models/aeon3_social02_labelled_topdown_top.centered_instance_multiclass" \
    --output "/ceph/aeon/aeon/code/scratchpad/sleap/multi_point_tracking/multi_animal_CameraNSEW/predictions/aeon3_social02_EVAL_frames_ceph_pr.slp" \
    --batch_size 16 \
    --only-labeled-frames \
    "/ceph/aeon/aeon/code/scratchpad/sleap/multi_point_tracking/multi_animal_CameraNSEW/aeon3_social02_EVAL_frames_ceph.slp"
