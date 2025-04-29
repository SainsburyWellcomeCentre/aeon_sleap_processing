# Step 2: Combine ID and Pose SLEAP Predictions
Use the python script [`combine_id_pose_chunk.py`](combine_pose_id_chunk.py) and its wrapper [`combine_id_pose_automated.py`](combine_pose_id_automated.py).

## What the code does
`combine_pose_id_chunk.py` combines identity and pose tracking data from SLEAP for a specified chunk.

The process:
1. Uses the Aeon API to load pose data (data register 212) from the top camera and identity data (data register 212) from both top camera and quadrant cameras (North, South, East, West)
2. Uses homography matrices to project quadrant camera coordinates to the top camera view
3. For each timestamp:
    - Matches identity detections to pose detections using a cost matrix:
	    - Each pose anchor point is matched to the nearest identity detection
		- The assignment maximizes the identity likelihood while minimizing spatial distance
		- A maximum distance threshold ensures only reasonable matches are made
    - Matches identity detections to pose detections based on spatial proximity
    - Uses a linear sum assignment algorithm to optimally assign identities to poses
4. Processes data in minute-by-minute chunks using parallel processing for efficiency
5. Outputs binary files with the combined ID-pose data (data register 222)

`combine_pose_id_automated.py` is a wrapper for `combine_pose_id_chunk.py` that enables batch processing of multiple chunks.

The process:
1. Scans for epoch directories between specified start and end times
2. Creates SLURM job arrays to process each chunk within the epochs
3. Monitors for new chunks and submits jobs for them automatically
4. Sends email notifications if any processing errors occur

## Running the code
For automated batch processing, activate your Aeon environment and run `combine_pose_id_automated.py` from the HPC login node with the following arguments:
- `root`: Path to the root directory where the SLEAP ID and pose data is, for finding the chunks to process
- `start`: Social period start time in format YYYY-MM-DDTHH-MM-SS
- `end`: Social period end time in format YYYY-MM-DDTHH-MM-SS
- `fps`: Frames per second of your cameras (default: 50)
- `output_dir`: Output directory for the processed data
- `job_id`: Optional job ID to match naming of already processed files
- `email`: Email address for error notifications

Example command:
```bash
python combine_pose_id_automated.py \
--root="/ceph/aeon/aeon/data/processed/AEON3/social0.2" \
--start="2024-02-09T15:00:00" \
--end="2024-02-09T17:00:00" \
--output_dir="/ceph/aeon/aeon/data/processed/AEON3/social0.2"
```
Update assumed file paths directly in the code as needed.

Note: at the moment, `combine_pose_id_automated.py` assumes you also have a top camera ID data. We have had top ID models previously but these should not be necessary in the future, hence why they are not mentioned in this pipeline. It should be easy to modify the code to remove this assumption.

## Next Steps
Your combined tracking data is ready for analysis!