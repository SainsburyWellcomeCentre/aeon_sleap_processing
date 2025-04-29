# Centered instance model training
If you prefer, you can train the centred instance model manually as you have for the centroid model, but as stated previously, we recommend using Optuna.

Use the python scripts [`sleap_optuna_train.py`](sleap_optuna_train.py), [`best_trial.py`](best_trial.py) and [`train_optimised_model.py`](train_optimised_model.py).

## What the code does
`sleap_optuna_train.py` launches a parallel Optuna search (via Submitit/SLURM) that trains many quick, partial centred-instance models (≈ 75 epochs) and scores them.

The process:
- Trains SLEAP centred instance model in parallel using the labelled datasets created in [Step 3](../../step3_SLEAP_labelling/README.md) and Optuna-suggested hyper-parameters (crop_size, initial_learning_rate, max_stride, filters, output_stride, num_fc_units, class_vectors_loss_weight)
    - Note that the first trial is an enqueued sensible seed trial with a combination of reasonable parameters to put Optuna on the right track
- Uses the newly trained partial centred instance model to make predictions (in conjunction with the manually trained centroid model) on a validation set and score the trial (see how the results are scored [here](eval_metric.md))
- Logs all results into a shared SQLite DB 

`best_trial.py` is a tiny helper that connects to the same SQLite DB, prints the leaderboard (or top-_N_) and the full hyper-parameter set for each winning trial.

`train_optimised_model.py` fetches the best (or top-_N_) parameter sets, then trains full centred-instance models (up to 600 epochs) and saves each one to `--model-output-dir`.

## Running the code
1. [Set up a SLEAP Optuna environment](setup_sleap_optuna_env.md)
2. [Set up SQLite](setup_sqlite.md)
3. Activate your SLEAP Optuna environment, `module load cuda/11.8` and run `sleap_optuna_train.py` from the HPC login node with the following arguments:
    - `study-path`: Full path to where the shared RDB should be created
    - `db-name`: Name of SQLite `.db` file; e.g. 'db.db'
    - `study-name`: Optuna study name
    - `labels-file`: Path to `.slp` labels created in [Step 3](../../step3_SLEAP_labelling/README.md)
    - `centroid-model-path`: Path to the centroid model for making predictions and calculating metrics
    - `slurm-output-dir`: Directory for the SLURM out and err files
    - `model-output-dir`: Directory to save partially trained models from each trial
    - `partition`: SLURM partition
    - `nodelist`: SLURM node
    - `n-tasks`: Number of parallel SLURM tasks
    - `n-trials`: Number of Optuna trials (i.e., number of different hyper-parameter combinations to try)
    - `slurm-job-name`: SLURM job name
    - `save-outputs`: Whether or not to save the partial sleap model and the SLEAP predictions generated for evaluation

Example command: 
```bash
python sleap_optuna_train.py \
--study-path "/ceph/aeon/aeon/code/scratchpad/sleap/multi_point_tracking/multi_animal_CameraNSEW/sleap_optuna_social_exps/databases" \
--db-name "a3s2.db" \
--slurm-output-dir "/ceph/aeon/aeon/code/scratchpad/sleap/multi_point_tracking/multi_animal_CameraNSEW/sleap_optuna_social_exps/log" \
--model-output-dir "/ceph/aeon/aeon/code/scratchpad/sleap/multi_point_tracking/multi_animal_CameraNSEW/sleap_optuna_social_exps/models" \
--labels-file "/ceph/aeon/aeon/code/scratchpad/sleap/multi_point_tracking/multi_animal_CameraNSEW/aeon3_social02_ceph.slp" \
--slurm-job-name "opt_a3s2" \
--study-name "opt_a3s2" \
--nodelist "gpu-sr675-34" &
```
Tips:
- Run this command inside a `tmux` session so that if your terminal closes, the Python process continues running safely in the background
- Append an ampersand `&` at the end of the command to immediately return control to your terminal without waiting for the script to finish
4. You can now view the top trials returned from `sleap_optuna_train.py`. This is optional, but is useful to give you an idea of how many full centered instance models to train. I.e., if the top 5 models all have very similar scores, you may want to train all 5 and use the best one as determined by evaluation in [Step 5](../../step5_SLEAP_evaluation/README.md). To view the top trials, run `best_trial.py` with the following arguments:
    - `study-name`: Optuna study name, should match the one you used in `sleap_optuna_train.py`
    - `db-path`: Path to the corresponding SQLite `.db` file

The code is currently configured to display the top 20 trials. Feel free to go in the code and change this number in the display_top_n_trials function call, or uncomment the line to display all trials.

Example command:
``` bash
python best_trial.py \
--study-name "opt_a3s2" \
--db-path "/ceph/aeon/aeon/code/scratchpad/sleap/multi_point_tracking/multi_animal_CameraNSEW/sleap_optuna/database/a3s2.db"
```
5. Now that the best combination(s) of hyper-parameters have been determined, you are ready to train the full SLEAP centered instance model(s). Run `train_optimised_model.py` with the following arguments:
    - `study-name`: Optuna study name, should match the one you used in `sleap_optuna_train.py`
    - `db-path`: Path to the corresponding SQLite `.db` file
    - `labels-file`:  Path to `.slp` labels created in [Step 3](../../step3_SLEAP_labelling/README.md)
    - `model-output-dir`: Directory to save fully trained model
    - `top-n`: Number of top trials to train a full SLEAP model for
    - `slurm-output-dir`: Directory for the SLURM out and err files
    - `slurm-job-name`: SLURM job name
    - `partition`: SLURM partition

Example command:
```bash
python train_optimised_model.py \
--study-name "opt_a3s2" \
--db-path "/ceph/aeon/aeon/code/scratchpad/sleap/multi_point_tracking/multi_animal_CameraNSEW/sleap_optuna/database/a3s2.db" \
--labels-file "/ceph/aeon/aeon/code/scratchpad/sleap/multi_point_tracking/multi_animal_CameraNSEW/aeon3_social02_ceph.slp" \
--model-output-dir "/ceph/aeon/aeon/code/scratchpad/sleap/multi_point_tracking/multi_animal_CameraNSEW/models" \
--top-n 1 \
--slurm-output-dir "/ceph/aeon/aeon/code/scratchpad/sleap/multi_point_tracking/multi_animal_CameraNSEW/slurm_output" \
--slurm-job-name "train_a3s2"
```

## Next Steps
Proceed to [Step 5](../../step5_SLEAP_evaluation/README.md)