# Step 1: Generate ID and Pose SLEAP Predictions
Use the Bonsai workflows [`InferIdentities_single_basic.bonsai`](bonsai_workflows/InferIdentities_single_basic.bonsai) and [`InferPoses_single_basic.bonsai`](bonsai_workflows/InferPoses_single_basic.bonsai) and their wrapper [`bonsai_sleap_automated.py`](bonsai_sleap_automated.py), followed by the [`bonsai_sleap_validate.ipynb`](bonsai_sleap_validate.ipynb) Jupyter Notebook.

## What the code does
`InferIdentities_single_basic.bonsai` and `InferPoses_single_basic.bonsai`are Bonsai workflow files that make SLEAP predictions using you trained and exported ID and pose SLEAP models. They output binary files containing the resulting tracking data (data register 202 for SLEAP ID data and 212 for SLEAP pose data).

`bonsai_sleap_automated.py` is a wrapper for the Bonsai workflows that enables batch processing of multiple chunks.

The process:
1. Identifies epoch directories between specified start and end times across multiple cameras (top camera and quadrant cameras)
2. Creates SLURM job arrays to process each chunk within the epochs
3. Monitors for new chunks and submits jobs for them automatically
4. Sends email notifications if any processing errors occur

Finally, `bonsai_sleap_validate.ipynb` is used to visualise and validate the tracking data resulting from the Bonsai workflows.

The process:
1. Loads processed pose data using the Aeon API
2. Randomly samples frames across multiple experiments to visually tracking quality
3. Overlays pose tracking points on the corresponding video frames
4. Creates interactive visualizations for manual review of tracking accuracy
5. Includes debugging capabilities for examining problematic frames

## Running the code
1. Install Bonsai for use on the HPC. Since the HPC recently switched from Ubuntu 20 to Ubuntu 24, the instructions for doing this are out of date and need to be revised. You can also use Bonsai locally, though the `bonsai_sleap_automated.py` wrapper would have to be adapted accordingly. In any case, the Bonsai and NuGet configs necessary for these workflows are provided in the `bonsai` folder. It should be noted that `InferIdentities_single_basic.bonsai` expects 
2. For automated batch processing, activate your Aeon environment and run `bonsai_sleap_automated.py` from the HPC login node with the following arguments:
	- `root`: Path to the root directory where the raw experiment data is, for finding the video files to process
	- `start`: Social period start time in format YYYY-MM-DDTHH-MM-SS
	- `end`: Social period end time in format YYYY-MM-DDTHH-MM-SS
	- `output_dir`: Output directory for the processed data
	- `partition`: SLURM partition
	- `job_id`: Optional job ID to match naming of already processed files
	- `email`: Email address for error notifications

Example command:
```bash
python bonsai_sleap_automated.py \
--root "/ceph/aeon/aeon/data/raw/AEON3/social0.2" \
--start="2024-02-09T15:00:00" \
--end="2024-02-09T17:00:00" \
--output_dir="/ceph/aeon/aeon/data/processed" \
--partition="gpu"
```
Update assumed file paths directly in the code as needed.

Note: at the moment, `bonsai_sleap_automated.py` assumes you also have a top camera ID model in addition to the quadrant camera ID models. We have had top ID models previously but these should not be necessary in the future, hence why they are not mentioned in this pipeline. It should be easy to modify the code to remove this assumption.

3. After processing is complete, use `bonsai_sleap_validate.ipynb` to validate results. Activate your Aeon environment, set the correct paths and times, and run the cells.

## Next Steps
Proceed to [Step 2](../step2_combining_SLEAP_predictions/README.md).