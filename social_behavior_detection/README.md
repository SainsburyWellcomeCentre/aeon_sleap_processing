# Social Behavior Detection

## Overview

This directory extends the Aeon SLEAP processing pipeline with automated detection of social behaviors in mice by analyzing combined pose and identity tracking data. Current functionality includes detection of fight events and tube test behaviors.

## What the code does
The behavior detection functions are found in `social_behaviour_detection.py`. 

`detect_tube_tests(root, start, end, parameters, skeleton, video_config)`
Detects social encounters and retreating in the arena corridors (similar to tube tests, since the mice cannot easily cross paths in the narrow corridor). For a detailed description of how this works, see [tube_test_detection_logic.md](tube_test_detection_logic.md)

`detect_fights(root, start, end, parameters, skeleton, video_config)`
Detects aggression bouts (fights) between mice. For a detailed description of how this works, see [fight_detection_logic.md](fight_detection_logic.md)

`detect_tube_tests.py` and `detect_fights.py` are wrappers for calling these functions and saving the results. `detect_tube_tests_all_epochs.py` and `detect_fights_all_epochs.py` are higher level wrappers that enable batch processing all epochs of an experiment.

The process:
1. Identifies epoch directories between specified start and end times using video file timestamps
2. Calculates epoch end times by examining video durations
3. Optionally splits epochs into 24-hour periods using configurable boundary hours for parallel processing (disadvantage is that it cuts what is technically continuous videos, so you could miss events on the edges)
4. Creates SLURM job arrays to process each epoch within the user specified start and end times
5. Sends email notifications if any processing errors occur

## Running the code
1. Activate your Aeon environment and run `detect_tube_tests_all_epochs.py`/`detect_fights_all_epochs.py` from the HPC login node with the following arguments:
    - `--root`: Path to the root directory where the experiment data is stored 
    - `--start`: Social period start time in format YYYY-MM-DDTHH-MM-SS 
    - `--end`: Social period end time in format YYYY-MM-DDTHH-MM-SS 
    - `--output_dir`: Output directory for CSV detection results
    - `--parameters`: Detection parameters as JSON string (optional, uses defaults)
    - `--skeleton`: Skeleton node mapping as JSON string (optional, uses defaults)
      - Tube test detection requires 4 key points: `"nose"`, `"head"`, `"centroid"`, `"tail_base"` → mapped to SLEAP parts like `"nose"`, `"head"`, `"spine2"`, `"spine4"`
      - Fight detection** requires 8 key points: all tube test points plus `"right_ear"`, `"left_ear"`, `"upper_spine"`, `"lower_spine"` → mapped to corresponding SLEAP parts
      - The mapping allows the algorithms to work with different SLEAP model naming conventions. For example, if your model uses "snout" instead of "nose", you'd set `"nose": "snout"` in the skeleton parameter.
    - `--video_config`: Video generation configuration as JSON string (optional, by default does not generate videos of the detected behaviors but can be useful for checking the quality of the results)
    - `--email`: Email address for error notifications 
    - `--boundary_hour`: Hour (0-23) for daily boundary when parallelizing into 24-hour chunks (optional, processes entire epochs if not specified)

Example command:
```bash
python detect_tube_tests_all_epochs.py 
--root=/ceph/aeon/aeon/data/ingest/AEON3/social0.4 \ 
--start="2024-08-28T09:00:00" \ 
--end="2024-09-09T13:00:00" \ 
--output_dir=/ceph/aeon/aeon/code/scratchpad/behaviour_detections/tube_tests
```
Update assumed file paths directly in the code as needed.

2. After processing is complete you should end up with one csv per epoch:
    - Fight detection output:
      - `start_timestamp`: Start time of detected fight
      - `end_timestamp`: End time of detected fight
      - `duration (seconds)`: Duration of the fight in seconds
      - `fight_end_x`: X coordinate of fight end location (mean position of both mice)
      - `fight_end_y`: Y coordinate of fight end location (mean position of both mice)
    - Tube test detection output:
      - `start_timestamp`: Start time of detected tube test
      - `end_timestamp`: End time of detected tube test
      - `winner_identity`: Identity of the mouse held its ground (i.e., won the tube test encounter)