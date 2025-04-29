#!/bin/bash
#SBATCH --job-name=make_composite_vid
#SBATCH --partition=gpu                             # partition (queue)
#SBATCH --gres=gpu:1                                # number of gpus per node
#SBATCH --nodes=1                                   # node count
#SBATCH --ntasks=1                                  # total number of tasks across all nodes
#SBATCH --mem=16G                                   # total memory per node 
#SBATCH --time=0-24:00:00                           # total run time limit (DD-HH:MM:SS)
#SBATCH --output=make_composite_vid_%A_%a.out
#SBATCH --array=0-0                                 # Array index, replace N with number of jobs - 1

# Define arrays with file-specific parameters
# experiments=("social0.2" "social0.2") # Add more experiments as needed

# arenas=("AEON4" "AEON4") # Add corresponding arenas

# dj_experiment_names=("social0.2-aeon4" "social0.2-aeon4") # Add corresponding DJ experiment names

# chunk_starts=("2024-02-28 15:00:00" "2024-02-25 18:00:00") # Add corresponding start times

#----

# experiments=("social0.3" "social0.3" "social0.3" "social0.3" "social0.3") # Add more experiments as needed

# arenas=("AEON3" "AEON3" "AEON3" "AEON3" "AEON3") # Add corresponding arenas

# dj_experiment_names=("social0.3-aeon3" "social0.3-aeon3" "social0.3-aeon3" "social0.3-aeon3" "social0.3-aeon3") # Add corresponding DJ experiment names

# chunk_starts=("2024-06-10 15:00:00" "2024-07-07 16:00:00" "2024-06-19 14:00:00" "2024-07-10 11:00:00" "2024-07-14 11:00:00") # Add corresponding start times

#----
# experiments=("social0.3" "social0.3" "social0.3" "social0.3" "social0.3") # Add more experiments as needed

# arenas=("AEON4" "AEON4" "AEON4" "AEON4" "AEON4") # Add corresponding arenas

# dj_experiment_names=("social0.3-aeon4" "social0.3-aeon4" "social0.3-aeon4" "social0.3-aeon4" "social0.3-aeon4") # Add corresponding DJ experiment names

# chunk_starts=("2024-06-09 13:00:00" "2024-07-04 14:00:00" "2024-06-19 14:00:00" "2024-07-08 13:00:00" "2024-07-13 10:00:00") # Add corresponding start times

#----
# experiments=("social0.4" "social0.4" "social0.4" "social0.4" "social0.4") # Add more experiments as needed

# arenas=("AEON3" "AEON3" "AEON3" "AEON3" "AEON3") # Add corresponding arenas

# dj_experiment_names=("social0.4-aeon3" "social0.4-aeon3" "social0.4-aeon3" "social0.4-aeon3" "social0.4-aeon3") # Add corresponding DJ experiment names

# chunk_starts=("2024-08-16 17:00:00" "2024-08-20 11:00:00" "2024-08-28 14:00:00" "2024-09-09 18:00:00" "2024-09-17 14:00:00") # Add corresponding start times

#----
# experiments=("social0.4" "social0.4" "social0.4" "social0.4" "social0.4") # Add more experiments as needed

# arenas=("AEON4" "AEON4" "AEON4" "AEON4" "AEON4") # Add corresponding arenas

# dj_experiment_names=("social0.4-aeon4" "social0.4-aeon4" "social0.4-aeon4" "social0.4-aeon4" "social0.4-aeon4") # Add corresponding DJ experiment names

# chunk_starts=("2024-08-16 17:00:00" "2024-08-20 11:00:00" "2024-08-28 14:00:00" "2024-09-09 16:00:00" "2024-09-17 13:00:00") # Add corresponding start times

#----

# experiments=("social0.4" "social0.4" "social0.2" "social0.2" "social0.3") # Add more experiments as needed

# arenas=("AEON3" "AEON3" "AEON4" "AEON4" "AEON4" ) # Add corresponding arenas

# dj_experiment_names=("social0.4-aeon3" "social0.4-aeon3" "social0.2-aeon4" "social0.2-aeon4" "social0.3-aeon4") # Add corresponding DJ experiment names

# chunk_starts=("2024-09-13 09:00:00" "2024-09-22 09:00:00" "2024-02-05 17:00:00" "2024-02-28 11:00:00" "2024-06-30 11:00:00") # Add corresponding start times


# experiments=("social0.2" "social0.3" "social0.4" "social0.2" "social0.3" "social0.4") # Add more experiments as needed

# arenas=("AEON3" "AEON3" "AEON3" "AEON4" "AEON4" "AEON4") # Add corresponding arenas

# dj_experiment_names=("social0.2-aeon3" "social0.3-aeon3" "social0.4-aeon3" "social0.2-aeon4" "social0.3-aeon4" "social0.4-aeon4") # Add corresponding DJ experiment names

# chunk_starts=("2024-02-17 17:00:00" "2024-07-05 12:00:00" "2024-09-08 11:00:00" "2024-02-17 12:00:00" "2024-07-02 13:00:00" "2024-09-08 11:00:00")

experiments=("social0.2") 

arenas=("AEON3") 

dj_experiment_names=("social0.2-aeon3") 

chunk_starts=(""2024-02-17 17:00:00"") 

# Get the index for the current job
index=${SLURM_ARRAY_TASK_ID}

# Run the Python script with corresponding parameters for the current job
python generate_composite_quadrant_videos.py \
    --experiment ${experiments[$index]} \
    --arena ${arenas[$index]} \
    --dj_experiment_name ${dj_experiment_names[$index]} \
    --dj_chunk_start "${chunk_starts[$index]}"