# Step 3: SLEAP Labelling
Use the Jupyter notebook [`composite_video_sleap_labelling_and_evaluating.ipynb`](composite_video_sleap_labelling_and_evaluating.ipynb).

## What the Code Does
This code creates labeled datasets for training SLEAP identity models, using the composite videos generated in [Step 2](../step2_composite_videos/README.md).

The process:
1. Extracts frames from composite quadrant camera videos, ensuring uniform spatial sampling across the arena
2. Transforms coordinates between top camera and quadrant camera views using the homography matrices from [Step 1](../step1_mapping/README.md)
3. Generates SLEAP dataset (`.slp`) files with labelled single points (centroids) for each animal
4. Prepares evaluation datasets for model validation

## Running the code
Activate your Aeon environment.
1. Configure two session dictionaries - one for training and one for evaluation - for your experiment:
	- Set the correct root directory paths to your data
	- Define time periods for single-animal recordings (both dictionaries) and multi-animal recordings (only training dictionary)
	- Specify working directory for output files
2. Create training data:
	- Generate dataset using `create_session_dataset()`
	- Check the position histograms to ensure good arena coverage
	- Adjust sampling parameters if needed to obtain ~1000 frames per animal (you may have to loop through the different cells a few times, hence the all_subj_data  copies and re-displaying the overview plots)
	- Export to CSV file
3. Transform coordinates from top camera to quadrant cameras:
	- Load homography matrices for each quadrant camera
	- Apply coordinate transformations using `transform_coordinates()`
	- Export to CSV file
4. Create evaluation data:
	- Generate dataset using `create_fully_labelled_dataset()
	- Export to CSV file
5. Transform coordinates from top camera to quadrant cameras: same process as bullet point 3.

Switch to your SLEAP environment:

6. Create SLEAP training dataset:
	- Rerun the code cell defining the session dictionary 
	- Generate a .slp training file from the saved CSV using `generate_slp_dataset()`

Switch to SLEAP GUI:

7. Manual annotation
	- Open the `.slp` file in the SLEAP GUI
	- Go through all the videos and adjust any labels that are not on the animal (this happens since the homography matrices aren’t perfect)
	- Save the labelled dataset with “\_labelled” suffix in the file name

Switch back to Jupyter notebook with SLEAP environment activated:

8. Update the video file paths in the .slp training file
	- Load the labelled data into the notebook
	- Update the video file paths to ceph file paths for training on the HPC
	- Save the data with the updated paths with the “\_ceph” suffix instead of “\_labelled”
9. Create SLEAP evaluation dataset:
	- Generate a .slp evaluation file for the quadrant cameras and top camera videos from the saved CSVs using `generate_slp_dataset()`

## Next Steps
Ignore the rest of the notebook for now and proceed to [Step 4](../step4_SLEAP_training/README.md).