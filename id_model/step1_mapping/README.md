# Step 1: Calculate Quadrant Camera to Top Camera Mapping

Use the Jupyter notebook [`quad_to_top_cam_mapping.ipynb`](quad_to_top_cam_mapping.ipynb).

## What the Code Does
This code creates pixel mappings (homography matrices) between each quadrant camera and the top camera. 

The process:
1. Randomly samples frames from video pairs
2. Detects and matches visual features between each quadrant camera and the top camera
3. Calculates homography matrices that transform quadrant camera views to match the top view
4. Validates the homographies against quality criteria (perspective, rotation, shear)
5. Averages good homographies and generates pixel mapping arrays
6. Saves the results for future use in spatial alignment

## Running the code
Activate your Aeon environment and modify the following variables as necessary:
 - Experiment settings: used for finding the AVI video files
    - `experiment`: Name of the experiment (e.g., 'social0.2')
    - `arena`: Arena identifier (e.g., 'AEON3'), equivalent to the acquisition computer name
    - `run_start`: Epoch name that leads to the video chunk you are interested in using
    - `chunk_start`: Timestamp of video chunk to use 
- Results directory:
    - `results_dir`: Where to save the mapping results
- Processing parameters:
    - `num_good_frames`: Number of valid homographies to collect (default: 50)
    - `perspective_threshold`, `shear_threshold`, `tolerance`: Threshold values in validation functions (default values should work fine)

## Next Steps
Proceed to [Step 2](../step2_composite_videos/README.md).