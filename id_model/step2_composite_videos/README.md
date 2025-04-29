# Step 2: Generate Composite Quadrant Camera Videos 
Use the python script [`generate_composite_quadrant_videos.py`](generate_composite_quadrant_videos.py) and the corresponding bash script [`generate_composite_quadrant_videos.sh`](generate_composite_quadrant_videos.sh).

## What the Code Does
The python script creates composite videos that automatically switch between quadrant cameras based on where the mice are in the arena. These videos will subsequently be labelled and used to train the SLEAP ID model, or will be used for evaluation. You are going to want to generate at least:
- Two 1h composite video for each animal on it’s own in the arena (one for training, one for evaluation)
- A 1h composite video for both animals together in the arena (for training)

The process:
1. Uses the homography matrices created in [Step 1](../step1_mapping/README.md) to map between camera views
2. Uses the mice’s positions from the pose model, either through loading with the aeon api or through the DataJoint database
3. Determines which quadrant camera has the best view of the mice for each frame
4. Automatically switches between quadrant cameras as mice move around the arena
5. Fills small gaps in tracking data to ensure smooth video transitions
6. Exports a single composite video where the view follows the mice

## Running the code
The bash script `generate_composite_quadrant_videos.sh` calls the python script with the following arguments:
- `experiments`: Names of the experiments (e.g., 'social0.2')
- `arenas`: Arena identifiers (e.g., 'AEON3'), equivalent to the acquisition computer names
- `dj_experiment_names`: Combined experiment-arena names (e.g., 'social0.2-aeon3')
- `chunk_starts`: Timestamps of video chunks to process (format: "YYYY-MM-DD HH:MM:SS")
The python script `generate_composite_quadrant_videos.py` should be ready to use, but you may be interested in adjusting the following:
- `TIMESTAMP_ERROR_TOLERANCE`: Allowable timestamp mismatch (default: 9 milliseconds)
- `MAX_GAP_TO_FILL`: Maximum gap in tracking to fill (default: 15 seconds)
- `VIDEO_EXPORT_DIR`: Where to save the output videos

Make sure to update the SLURM `--array` setting based on how many chunk starts you have. Activate your Aeon environment and run the bash script on the HPC with the command `sbatch generate_composite_quadrant_videos.sh`.

## Next Steps 
Proceed to [Step 3](../step3_SLEAP_labelling/README.md).