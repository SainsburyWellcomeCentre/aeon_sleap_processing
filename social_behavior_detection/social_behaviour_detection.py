import swc.aeon
from swc.aeon.io import api
from aeon.schema.schemas import social02
from aeon.io import video
from dotmap import DotMap
import re
import os
import warnings
import pandas as pd
import numpy as np
import datetime
import importlib


def detect_tube_tests(
    root: str,
    start: datetime,
    end: datetime,
    parameters: dict | None = None,
    skeleton: dict | None = None,
    video_config: dict | None = None,
) -> pd.DataFrame:
    """Detects tube tests in the video.

    Args:
        root (str): The root directory of the full pose SLEAP data.
        start (datetime): The left bound of the time range to extract.
        end (datetime): The right bound of the time range to extract.
        parameters (dict, optional): The parameters of the detection.
        skeleton (dict, optional): A mapping of the required nodes (nose, head, centroid, and tail_base) to their corresponding names in your SLEAP project.
        video_config (dict, optional): The configuration for generating videos.

    Returns:
        DataFrame: A pandas data frame containing the detected tube tests with start_timestamp, end_timestamp, and winner_identity columns.
    """
    parameter_defaults = {
        "angle_tolerance": 45,
        "max_distance_start": 50,
        "max_frame_gap": 20,
        "min_tube_test_start_frames": 15,
        "search_window_seconds": 1,
        "min_distance": 30,
        "max_distance_end": 60,
        "movement_threshold": 2,
        "gate_width": 20
    }
    allowed_parameters_keys = set(parameter_defaults.keys())
    if parameters is None:
        parameters = {}
    else:
        for key in parameters.keys():
            if key not in allowed_parameters_keys:
                raise ValueError(f"Invalid parameter key: {key}. Allowed keys are: {allowed_parameters_keys}")
    for key, value in parameter_defaults.items():
        parameters.setdefault(key, value)

    skeleton_defaults = {
        "nose": "nose",
        "head": "head", 
        "centroid": "spine2",
        "tail_base": "spine4"
    }
    allowed_skeleton_keys = set(skeleton_defaults.keys())
    if skeleton is None:
        skeleton = {}
    else:
        for key in skeleton.keys():
            if key not in allowed_skeleton_keys:
                raise ValueError(f"Invalid skeleton key: {key}. Allowed keys are: {allowed_skeleton_keys}")
    for key, value in skeleton_defaults.items():
        skeleton.setdefault(key, value)

    video_config_defaults = {
        "gen_vids": False,
        "video_save_path": None,
        "camera": "CameraTop"
    }
    allowed_video_config_keys = set(video_config_defaults.keys())
    if video_config is None:
        video_config = {}
    else:
        for key in video_config.keys():
            if key not in allowed_video_config_keys:
                raise ValueError(f"Invalid video_config key: {key}. Allowed keys are: {allowed_video_config_keys}")
    for key, value in video_config_defaults.items():
        video_config.setdefault(key, value)
    
    # Find the CameraTop frame rate
    metadata_root = root.replace("ingest", "raw")
    experiment = re.split(r'[\\/]', root)[7]
    experiment = re.sub(r'[./\\]', '', experiment)
    experiment = "social02" # DELETE
    schemas_module = importlib.import_module(f"aeon.schema.schemas")
    experiment_schema = getattr(schemas_module, experiment, None)
    metadata_reader = getattr(experiment_schema, "Metadata", None)
    metadata = api.load(metadata_root, metadata_reader)['metadata'].iloc[0]
    trigger_freq = metadata.Devices.CameraTop.TriggerFrequency
    fps = int(getattr(metadata.Devices.VideoController, f'{trigger_freq}', None))

    # LOAD DATA
    reader = swc.aeon.io.reader.Pose(pattern="CameraTop_222*")
    print(f"Loading data from {root} with pattern CameraTop_222*, start: {start}, end: {end}")
    tracks_df = api.load(root=root, reader=reader, start=start, end=end)
    experiment_times = get_experiment_times(metadata_root, start, end)
    print(f"tracks_df = api.load(root={root}, reader=reader, start={start}, end={end})")
    tracks_df = exclude_maintenance_data(tracks_df, experiment_times)
    print(tracks_df.shape)
    print(tracks_df.head())
    print(tracks_df["model"].unique())
    
    if len(tracks_df['identity'].unique()) != 2:
        raise ValueError("There should be exactly 2 unique identities in the data.")

    # CONVERT TRACKS DATAFRAME TO NUMPY ARRAY
    tracks_df.reset_index(inplace=True)
    # Preprocess
    tracks_df['time'] = pd.to_datetime(tracks_df['time'])
    tracks_df.sort_values(by=['time', 'identity', 'part'], inplace=True)
    # Convert identities to numeric codes (0 and 1)
    identity_mapping = {id_val: idx for idx, id_val in enumerate(tracks_df['identity'].unique())}
    tracks_df['identity_code'] = tracks_df['identity'].map(identity_mapping)
    
    # Store the original identities for later use
    original_identities = list(identity_mapping.keys())
    
    # Get the absolute first and last timestamp in the data
    min_time = tracks_df['time'].min()
    max_time = tracks_df['time'].max()
    
    # Create mapping for parts to indices
    part_to_index = pd.Series(index=[skeleton['nose'], skeleton['head'], skeleton['centroid'], skeleton['tail_base']], data=np.arange(4))
    
    # Filter to keep only required parts
    tracks_df = tracks_df[tracks_df['part'].isin(part_to_index.index)]
    tracks_df['part_id'] = tracks_df['part'].map(part_to_index)
    
    # Create deterministic frame index based on timestamp
    tracks_df['frame_id'] = ((tracks_df['time'] - min_time).dt.total_seconds() * fps).round().astype(int)
    
    # Determine dimensions
    num_frames = tracks_df['frame_id'].max() + 1
    num_parts = 4
    num_mice = 2  
    num_coordinates = 2
    
    # Initialize the array
    tracks = np.full((num_mice, num_coordinates, num_parts, num_frames), np.nan)
    
    # Use numpy advanced indexing to populate the array
    mouse_ids = tracks_df['identity_code'].values
    part_ids = tracks_df['part_id'].values
    frame_ids = tracks_df['frame_id'].values
    x_coords = tracks_df['x'].values
    y_coords = tracks_df['y'].values

    # Create a mapping from frame_id to original timestamp
    # We'll use the first occurrence of each frame_id to get a consistent timestamp
    frame_to_timestamp = {}
    for _, group_df in tracks_df.groupby('frame_id'):
        frame_id = group_df['frame_id'].iloc[0]
        timestamp = group_df['time'].iloc[0]
        frame_to_timestamp[frame_id] = timestamp
    # Store this mapping for later use
    frame_timestamps = pd.Series(frame_to_timestamp)
    
    # Populate the tracks array
    tracks[mouse_ids, 0, part_ids, frame_ids] = x_coords
    tracks[mouse_ids, 1, part_ids, frame_ids] = y_coords

    # CALCULATIONS AND EXTRACTION OF POSSIBLE TUBE TEST FRAMES 
    # Centroid distances 
    centroid_mouse0 = tracks[0, :, part_to_index[skeleton['centroid']], :]
    centroid_mouse1 = tracks[1, :, part_to_index[skeleton['centroid']], :]
    centroid_distances = np.linalg.norm(centroid_mouse0 - centroid_mouse1, axis=0)

    # Relative spine distances
    spine4_mouse0 = tracks[0, :, part_to_index[skeleton['tail_base']], :]
    head_mouse0 = tracks[0, :, part_to_index[skeleton['head']], :]
    head_mouse1 = tracks[1, :, part_to_index[skeleton['head']], :]
    relative_distances = np.zeros((2, tracks.shape[3]))
    relative_distances[0, :] = np.linalg.norm(spine4_mouse0 - head_mouse0, axis=0)
    relative_distances[1, :] = np.linalg.norm(spine4_mouse0 - head_mouse1, axis=0)

    # Extremity distances
    spine4_mouse1 = tracks[1, :, part_to_index[skeleton['tail_base']], :]
    extremity_distances = np.zeros((4, tracks.shape[3]))
    extremity_distances[0, :] = np.linalg.norm(head_mouse0 - head_mouse1, axis=0)
    extremity_distances[1, :] = np.linalg.norm(spine4_mouse0 - spine4_mouse1, axis=0)
    extremity_distances[2, :] = np.linalg.norm(spine4_mouse0 - head_mouse1, axis=0)
    extremity_distances[3, :] = np.linalg.norm(spine4_mouse1 - head_mouse0, axis=0)

    # Orientation
    # Calculate differences in x and y coordinates
    dy_tail_nose = tracks[:, 1, part_to_index[skeleton['nose']], :] - tracks[:, 1, part_to_index[skeleton['tail_base']], :]
    dy_tail_head = tracks[:, 1, part_to_index[skeleton['head']], :] - tracks[:, 1, part_to_index[skeleton['tail_base']], :]
    dx_tail_nose = tracks[:, 0, part_to_index[skeleton['nose']], :] - tracks[:, 0, part_to_index[skeleton['tail_base']], :]
    dx_tail_head = tracks[:, 0, part_to_index[skeleton['head']], :] - tracks[:, 0, part_to_index[skeleton['tail_base']], :]
    # Calculate angles: 0 degrees if the mice are facing towards the nest, angles increase counterclockwise
    angles_tail_nose = np.degrees(np.arctan2(-dy_tail_nose, dx_tail_nose))
    angles_tail_head = np.degrees(np.arctan2(-dy_tail_head, dx_tail_head))
    # Adjust angles to be in the range [0, 360)
    angles_tail_nose = np.where(angles_tail_nose < 0, angles_tail_nose + 360, angles_tail_nose)
    angles_tail_head = np.where(angles_tail_head < 0, angles_tail_head + 360, angles_tail_head)
    # When angles_tail_nose is NaN, use angles_tail_head
    orientations = np.where(np.isnan(angles_tail_nose), angles_tail_head, angles_tail_nose)

    # Adjust the orientation of mouse 2
    adjusted_orientations = (orientations[1] + 180) % 360

    # Condition 1: the mice have opposite orientations, within a certain tolerance
    orientation_condition = np.isclose(orientations[0], adjusted_orientations, atol=parameters['angle_tolerance'])
    # Condition 2: the distance between the mice's centroids is less than a certain threshold, ensuring they are close to each other
    distance_condition = centroid_distances < parameters['max_distance_start']
    # Condition 3: relative spine measure, removes cases where mice are side by side
    relative_distance_condition = relative_distances[1] > relative_distances[0]
    # Condition 4: the mice's tail-to-tail distance is greater than their nose-to-nose distance, removes cases where mice are back-to-back
    extremity_distance_condition = extremity_distances[1] > extremity_distances[0]
    # Find frames where all conditions are true
    possible_tube_test_starts = np.where(np.logical_and.reduce([orientation_condition, distance_condition, relative_distance_condition, extremity_distance_condition]))[0]

    # FILTER POSSIBLE TUBE TEST FRAMES TO ONLY KEEP THOSE WHERE THE MICE ARE BOTH WITHIN THE CORRIDOR ROI
    inner_radius = float(metadata.ActiveRegion.ArenaInnerRadius)
    outer_radius = float(metadata.ActiveRegion.ArenaOuterRadius)
    center_x = float(metadata.ActiveRegion.ArenaCenter.X)
    center_y = float(metadata.ActiveRegion.ArenaCenter.Y)
    nest_y1 = float(metadata.ActiveRegion.NestRegion.ArrayOfPoint[1].Y)
    nest_y2 = float(metadata.ActiveRegion.NestRegion.ArrayOfPoint[2].Y)
    # for every key containing the words Gate and Rfid in metadata.Devices.keys()
    gate_coordinates = []
    for key in metadata.Devices.keys():
        if 'Gate' in key and 'Rfid' in key:
            gate_coordinates.append(metadata.Devices[key].Location)

    # Create an array of frame numbers
    frame_numbers = np.arange(tracks.shape[3])

    # Get the x and y coordinates of spine2 for both mice
    spine2_x = tracks[:, 0, part_to_index[skeleton['centroid']], :]
    spine2_y = tracks[:, 1, part_to_index[skeleton['centroid']], :]

    # Calculate the squared distance from the center of the ROI
    dist_squared_from_centre = (spine2_x - center_x)**2 + (spine2_y - center_y)**2

    # Check if the distance is within the squared radii for both mice
    within_roi = (inner_radius**2 <= dist_squared_from_centre) & (dist_squared_from_centre <= outer_radius**2)

    # Check if the mice are in excluded regions
    in_excluded_region_nest = (spine2_x > center_x) & ((spine2_y >= nest_y1) & (spine2_y <= nest_y2))
    in_excluded_region_entrance = np.zeros(spine2_x.shape, dtype=bool)
    for gate_coordinate in gate_coordinates:
        dist_squared_from_gate = (spine2_x - float(gate_coordinate.X))**2 + (spine2_y - float(gate_coordinate.Y))**2
        in_excluded_region_entrance = in_excluded_region_entrance | (dist_squared_from_gate <= parameters['gate_width']**2)
    # in_excluded_region_entrance2 = (spine2_x < center_x) & ((spine2_y >= entrance_y1) & (spine2_y <= entrance_y2))

    # Update the ROI condition to exclude the specified regions
    within_roi = within_roi & ~np.any(in_excluded_region_nest | in_excluded_region_entrance, axis=0)
    within_roi_both_mice = np.all(within_roi, axis=0)

    # Filter the frame numbers where both mice are within the ROI
    frame_numbers_in_roi = frame_numbers[within_roi_both_mice]

    # Filter possible tube test frames to only keep those where the mice are both within the corridor ROI
    possible_tube_test_starts = np.intersect1d(possible_tube_test_starts, frame_numbers_in_roi)

    # DIVIDE POSSIBLE TUBE TEST FRAMES INTO SUBARRAYS OF CONSECUTIVE FRAMES = ONE POSSIBLE TUBE TEST EVENT
    # Divide possible_tube_test_starts into sub_arrays of consecutive frames (allowing for gaps up to a certain max)
    diffs = np.diff(possible_tube_test_starts)
    indices = np.where(diffs > parameters['max_frame_gap'])[0]
    indices += 1
    possible_tube_test_starts = np.split(possible_tube_test_starts, indices)

    # Filter sub_arrays to keep only those with more than a certain number of frames
    possible_tube_test_starts = [sub_array for sub_array in possible_tube_test_starts if len(sub_array) > parameters['min_tube_test_start_frames']]

    # EXTRACTION OF TRUE TUBE TEST EVENTS
    tube_tests_data = {'start_timestamp': [], 'end_timestamp': [], 'winner_identity': []}
    for subarray in possible_tube_test_starts:
        # Check each possible_tube_test_starts frame interval for tracking errors
        # Skeleton flipping (i.e., tail end being mistaken for head) can lead to false tube test detections
        # Take all orientations in the interval, including frames that did not meet all the tube test start conditions
        all_start_orientations = orientations[:, subarray[0]:subarray[-1]+1]
        # Find how often the mice have the same orientation, within a certain tolerance
        orientation_condition = np.isclose(all_start_orientations[0], all_start_orientations[1], atol=parameters['angle_tolerance']) 
        count = np.count_nonzero(orientation_condition)
        # Move to the next possible tube test start if skeleton flipping is detected
        if count > 1:
            continue

        # Find the possible tube test end frames
        first_possible_start_frame = subarray[0]
        last_possible_start_frame = subarray[-1]
        search_window = int(np.ceil(fps*parameters['search_window_seconds']))

        # Condition 1: the mice have the same orientations, within a certain tolerance
        orientation_condition = np.isclose(orientations[0, last_possible_start_frame:last_possible_start_frame + search_window], orientations[1, last_possible_start_frame:last_possible_start_frame + search_window], atol=parameters['angle_tolerance'])
        # Condition 2: the distance between the mice's centroids is more than a certain threshold, removes cases where mice are fighting or side-by-side
        min_distance_condition = centroid_distances[last_possible_start_frame:last_possible_start_frame + search_window] > parameters['min_distance']
        # Condition 3: the distance between the mice's centroids is less than a certain threshold, removes cases where mice "teleport" due to tracking errors
        max_distance_condition = centroid_distances[last_possible_start_frame:last_possible_start_frame + search_window] < parameters['max_distance_end']
        # Find frames where all conditions are true
        possible_tube_test_ends = last_possible_start_frame + np.where(np.logical_and.reduce([orientation_condition, min_distance_condition, max_distance_condition]))[0]
        # If there are frames where end conditions are met, clean identity tracking and check addtional conditions reliant on identity
        if len(possible_tube_test_ends) > 0:
            # Make list of frames where the identities are swapped
            # Trim the centroid data to the frames we are currently considering
            centroid_mouse0_trimmed = centroid_mouse0[:,first_possible_start_frame:last_possible_start_frame + search_window]
            centroid_mouse1_trimmed = centroid_mouse1[:,first_possible_start_frame:last_possible_start_frame + search_window]
            # Initialize variables to hold the last known positions of each mouse (used to deal with NaN values in the tracking data)
            last_known_pos0 = centroid_mouse0_trimmed[:, 0]
            last_known_pos1 = centroid_mouse1_trimmed[:, 0]
            # Initialize a list to hold the frames where the identities are swapped
            id_swaps = []
            # Vote counter for identity assignment (similar to clean_swaps)
            track_votes = np.zeros((2, 2), dtype=np.int64)
            # Count first frame if valid
            if not np.isnan(centroid_mouse0_trimmed[:, 0]).any() and not np.isnan(centroid_mouse1_trimmed[:, 0]).any():
                track_votes[0, 0] += 1
                track_votes[1, 1] += 1
            # Loop over the frames from the second frame to the last
            for i in range(1, last_possible_start_frame + search_window - first_possible_start_frame):
                if np.isnan(centroid_mouse0_trimmed[:, i]).any() and np.isnan(centroid_mouse1_trimmed[:, i]).any():
                    continue
                # Calculate the Euclidean distance from each centroid in the current frame to each centroid in the previous frame
                dists = np.zeros((2, 2))
                dists[0, 0] = np.sqrt(np.sum((centroid_mouse0_trimmed[:, i] - last_known_pos0)**2))
                dists[0, 1] = np.sqrt(np.sum((centroid_mouse0_trimmed[:, i] - last_known_pos1)**2))
                dists[1, 0] = np.sqrt(np.sum((centroid_mouse1_trimmed[:, i] - last_known_pos0)**2))
                dists[1, 1] = np.sqrt(np.sum((centroid_mouse1_trimmed[:, i] - last_known_pos1)**2))
                if dists[0, 0] + dists[1, 1] <= dists[0, 1] + dists[1, 0]:
                    last_known_pos0 = centroid_mouse0_trimmed[:, i]
                    last_known_pos1 = centroid_mouse1_trimmed[:, i]
                    track_votes[0, 0] += 1
                    track_votes[1, 1] += 1
                else:
                    last_known_pos0 = centroid_mouse1_trimmed[:, i]
                    last_known_pos1 = centroid_mouse0_trimmed[:, i]
                    id_swaps.append(i)
                    track_votes[0, 1] += 1
                    track_votes[1, 0] += 1
            
            # Global identity correction via majority vote
            need_global_swap = track_votes[0, 1] > track_votes[0, 0]
            if need_global_swap:
                # Flip the id_swaps - frames that were locally swapped become unswapped and vice versa
                total_frames = last_possible_start_frame + search_window - first_possible_start_frame
                all_frames = set(range(total_frames))
                swapped_frames = set(id_swaps)
                id_swaps = list(all_frames - swapped_frames)

            # Find which mouse turned around (loser)
            orientations_cleaned = orientations[:, first_possible_start_frame:last_possible_start_frame + search_window].copy()
            orientations_cleaned[:, id_swaps] = orientations_cleaned[::-1, id_swaps]
            start_orientations = orientations_cleaned[:, subarray-first_possible_start_frame]
            start_orientations = np.nanmean(start_orientations, axis=1)
            end_orientations = orientations_cleaned[:, possible_tube_test_ends-first_possible_start_frame]
            end_orientations = np.nanmean(end_orientations, axis=1)
            loser_mouse_index = np.argmax(np.abs(start_orientations - end_orientations))
            winner_mouse_index = 1 - loser_mouse_index  # If loser is 0, winner is 1; if loser is 1, winner is 0
            
            # Condition 4: the loser is in front of the winner, removes cases where mouse A squeezes past mouse B, and mouse B turns around (false tube test detection)
            extremity_distances_cleaned = extremity_distances[:, first_possible_start_frame:last_possible_start_frame + search_window].copy()
            extremity_distances_cleaned[-2:, id_swaps] = extremity_distances_cleaned[-2:][::-1, id_swaps]
            mean_tail0_head1_distance = np.nanmean(extremity_distances_cleaned[2, possible_tube_test_ends-first_possible_start_frame])
            mean_tail1_head0_distance = np.nanmean(extremity_distances_cleaned[3, possible_tube_test_ends-first_possible_start_frame])
            front_mouse_condition = mean_tail0_head1_distance < mean_tail1_head0_distance if loser_mouse_index == 0 else mean_tail1_head0_distance < mean_tail0_head1_distance
            # Condition 5: the loser's average movement between each frame is larger than a certain threshold, removes cases where the mice are stationary (e.g., grooming)
            tracks_cleaned = tracks[:, :, :, first_possible_start_frame:last_possible_start_frame + search_window].copy()
            tracks_cleaned[:, :, :, id_swaps] = tracks_cleaned[::-1, :, :, id_swaps]
            points_frame = tracks_cleaned[loser_mouse_index, :, part_to_index[skeleton['centroid']], last_possible_start_frame-first_possible_start_frame:-1]  # all but the last frame
            points_next_frame = tracks_cleaned[loser_mouse_index, :, part_to_index[skeleton['centroid']], last_possible_start_frame-first_possible_start_frame+1:]  # all but the first frame
            differences = points_next_frame - points_frame
            mean_movement = np.nanmean(np.linalg.norm(differences, axis=0))
            movement_condition = mean_movement > parameters['movement_threshold']
            # Add tube test to final table if all end conditions are met
            if front_mouse_condition and movement_condition:
                start_timestamp = frame_timestamps.get(first_possible_start_frame, 
                                      min_time + pd.Timedelta(seconds=first_possible_start_frame/fps))
                tube_tests_data['start_timestamp'].append(start_timestamp)
                end_frame = possible_tube_test_ends[0]
                end_timestamp = frame_timestamps.get(end_frame, 
                                    min_time + pd.Timedelta(seconds=end_frame/fps))
                tube_tests_data['end_timestamp'].append(end_timestamp)
                # Add the identity of the winner mouse
                winner_identity = original_identities[winner_mouse_index]
                tube_tests_data['winner_identity'].append(winner_identity)
            
    tube_tests_df = pd.DataFrame(tube_tests_data)

    if video_config['gen_vids'] == True:
        generate_videos(root, tube_tests_df, video_config['camera'], video_config['video_save_path'], video_name_prefix="tube_test", padding=5)

    return tube_tests_df

def detect_fights(
    root: str,
    start: datetime,
    end: datetime,
    parameters: dict | None = None,
    skeleton: dict | None = None,
    video_config: dict | None = None,
) -> pd.DataFrame:
    """Detects fights in the video.

    Args:
        root (str): The root directory of the full pose SLEAP data.
        start (datetime): The left bound of the time range to extract.
        end (datetime): The right bound of the time range to extract.
        parameters (dict, optional): The parameters of the detection.
        skeleton (dict, optional): A mapping of the required nodes (nose, head, right_ear, left_ear, upper_spine, centroid, lower_spine, and tail_base) to their corresponding names in your SLEAP project.
        video_config (dict, optional): The configuration for generating videos.

    Returns:
        DataFrame: A pandas data frame containing the detected fights.
    """
    parameter_defaults = {
        "cm2px": 5.4, # 1 cm = 5.4 px
        "max_distance": 20,
        "max_nose_head_distance": 7,
        "max_interspinal_distance": 10,
        "min_blob_speed": 3,
        "max_frame_gap": 200,
        "min_num_frames": 5,
        "max_frame_gap_w_empty_frames": 100,
        "min_centroid_speed": 20, # cm/s min speed for fighting
        "min_both_centroid_speed": 15
    }
    allowed_parameters_keys = set(parameter_defaults.keys())
    if parameters is None:
        parameters = {}
    else:
        for key in parameters.keys():
            if key not in allowed_parameters_keys:
                raise ValueError(f"Invalid parameter key: {key}. Allowed keys are: {allowed_parameters_keys}")
    for key, value in parameter_defaults.items():
        parameters.setdefault(key, value)

    skeleton_defaults = {
        "nose": "nose",
        "head": "head", 
        "right_ear": "right_ear",
        "left_ear": "left_ear",
        "upper_spine": "spine1", # closest to the head
        "centroid": "spine2",
        "lower_spine": "spine3",
        "tail_base": "spine4"
    } 
    allowed_skeleton_keys = set(skeleton_defaults.keys())
    if skeleton is None:
        skeleton = {}
    else:
        for key in skeleton.keys():
            if key not in allowed_skeleton_keys:
                raise ValueError(f"Invalid skeleton key: {key}. Allowed keys are: {allowed_skeleton_keys}")
    for key, value in skeleton_defaults.items():
        skeleton.setdefault(key, value)

    video_config_defaults = {
        "gen_vids": False,
        "video_save_path": None,
        "camera": "CameraTop"
    }
    allowed_video_config_keys = set(video_config_defaults.keys())
    if video_config is None:
        video_config = {}
    else:
        for key in video_config.keys():
            if key not in allowed_video_config_keys:
                raise ValueError(f"Invalid video_config key: {key}. Allowed keys are: {allowed_video_config_keys}")
    for key, value in video_config_defaults.items():
        video_config.setdefault(key, value)
    
    # Find the CameraTop frame rate
    metadata_root = root.replace("ingest", "raw")
    experiment = re.split(r'[\\/]', root)[7]
    experiment = re.sub(r'[./\\]', '', experiment)
    experiment = "social02"
    schemas_module = importlib.import_module(f"aeon.schema.schemas")
    experiment_schema = getattr(schemas_module, experiment, None)
    metadata_reader = getattr(experiment_schema, "Metadata", None)
    metadata = api.load(metadata_root, metadata_reader)['metadata'].iloc[0]
    trigger_freq = metadata.Devices.CameraTop.TriggerFrequency
    fps = int(getattr(metadata.Devices.VideoController, f'{trigger_freq}', None))

    # LOAD DATA
    reader = swc.aeon.io.reader.Pose(pattern="CameraTop_222*")
    print(f"Loading data from {root} with pattern CameraTop_222*, start: {start}, end: {end}")
    tracks_df = api.load(root=root, reader=reader, start=start, end=end)
    experiment_times = get_experiment_times(metadata_root, start, end)
    tracks_df = exclude_maintenance_data(tracks_df, experiment_times)
    print(tracks_df.shape)
    print(tracks_df.head())
    print(tracks_df["model"].unique())
    
    if len(tracks_df['identity'].unique()) != 2:
        raise ValueError("There should be exactly 2 unique identities in the data.")
    
    # CONVERT TRACKS DATAFRAME TO NUMPY ARRAY
    tracks_df.reset_index(inplace=True)
    # Preprocess
    tracks_df['time'] = pd.to_datetime(tracks_df['time'])
    tracks_df.sort_values(by=['time', 'identity', 'part'], inplace=True)
    # Convert identities to numeric codes (0 and 1)
    identity_mapping = {id_val: idx for idx, id_val in enumerate(tracks_df['identity'].unique())}
    tracks_df['identity_code'] = tracks_df['identity'].map(identity_mapping)
    
    # Get the absolute first and last timestamp in the data
    min_time = tracks_df['time'].min()
    max_time = tracks_df['time'].max()
    
    # Create mapping for parts to indices
    part_to_index = pd.Series(index=[skeleton['nose'], skeleton['head'], skeleton['right_ear'], skeleton['left_ear'], 
                                     skeleton['upper_spine'], skeleton['centroid'], skeleton['lower_spine'], skeleton['tail_base']], 
                              data=np.arange(8))
    edge_inds = [[0,1], [1,2], [1,3], [1,4], [4,5], [5,6], [6,7]]
    
    # Filter to keep only required parts
    tracks_df = tracks_df[tracks_df['part'].isin(part_to_index.index)]
    tracks_df['part_id'] = tracks_df['part'].map(part_to_index)
    
    # Create deterministic frame index based on timestamp
    tracks_df['frame_id'] = ((tracks_df['time'] - min_time).dt.total_seconds() * fps).round().astype(int)
    
    # Determine dimensions
    num_frames = tracks_df['frame_id'].max() + 1
    num_parts = 8
    num_mice = 2  
    num_coordinates = 2
    
    # Initialize the array
    tracks = np.full((num_mice, num_coordinates, num_parts, num_frames), np.nan)
    
    # Use numpy advanced indexing to populate the array
    mouse_ids = tracks_df['identity_code'].values
    part_ids = tracks_df['part_id'].values
    frame_ids = tracks_df['frame_id'].values
    x_coords = tracks_df['x'].values
    y_coords = tracks_df['y'].values

    # Create a mapping from frame_id to original timestamp
    # We'll use the first occurrence of each frame_id to get a consistent timestamp
    frame_to_timestamp = {}
    for _, group_df in tracks_df.groupby('frame_id'):
        frame_id = group_df['frame_id'].iloc[0]
        timestamp = group_df['time'].iloc[0]
        frame_to_timestamp[frame_id] = timestamp
    # Store this mapping for later use
    frame_timestamps = pd.Series(frame_to_timestamp)
    
    # Populate the tracks array
    tracks[mouse_ids, 0, part_ids, frame_ids] = x_coords
    tracks[mouse_ids, 1, part_ids, frame_ids] = y_coords

    # LOAD BLOB DATA
    blob_root = root.replace("ingest", "raw")
    reader = swc.aeon.io.reader.Position(pattern="CameraTop_200*")
    centroid_blob_data = api.load(blob_root, reader, start, end)
    centroid_blob_data.reset_index(inplace=True)
    centroid_blob_data.dropna(inplace=True)
    
    # CALCULATIONS AND EXTRACTION OF FIGHTING FRAMES
    # Centroid distances 
    centroid_mouse0 = tracks[0, :, part_to_index[skeleton['centroid']], :]
    centroid_mouse1 = tracks[1, :, part_to_index[skeleton['centroid']], :]
    centroid_distances = np.linalg.norm(centroid_mouse0 - centroid_mouse1, axis=0)
    centroid_distances_ffill = pd.Series(centroid_distances).ffill().to_numpy()

    # Internode distances
    internode_distances_mouse0 = np.zeros((len(edge_inds), tracks.shape[3]))
    internode_distances_mouse1 = np.zeros((len(edge_inds), tracks.shape[3]))
    for i, node_pair in enumerate(edge_inds):
        internode_distances_mouse0[i] = np.linalg.norm(tracks[0, :, node_pair[0], :] - tracks[0, :, node_pair[1], :], axis=0)
        internode_distances_mouse1[i] = np.linalg.norm(tracks[1, :, node_pair[0], :] - tracks[1, :, node_pair[1], :], axis=0)
    nose_head_distances_mouse0 = internode_distances_mouse0[0,:]
    nose_head_distances_mouse1 = internode_distances_mouse1[0,:]
    mean_interspinal_distances_mouse0 = np.nanmean(internode_distances_mouse0[3:,:], axis=0)
    mean_interspinal_distances_mouse1 = np.nanmean(internode_distances_mouse1[3:,:], axis=0)

    # Blob speed
    dxy = centroid_blob_data[["x", "y"]].diff().values[1:]
    dt_raw = np.diff(centroid_blob_data["time"])
    # Convert timedelta to milliseconds for comparison (divide by 1e6 for microseconds, then by 1000 for ms)
    dt_ms = dt_raw.astype('timedelta64[ms]').astype(float)
    min_expected_dt_ms = 10.0  # 10 milliseconds (half the expected sampling interval)
    # Find indices where time differences are abnormally small
    abnormal_indices = np.where(dt_ms < min_expected_dt_ms)[0]
    if len(abnormal_indices) > 0:
        print(f"Found {len(abnormal_indices)} abnormally small time differences")
        # Get the corresponding indices in the original dataframe
        drop_indices = abnormal_indices + 1  # +1 because diff reduces array size by 1
        # Drop these rows from centroid_blob_data
        centroid_blob_data = centroid_blob_data.drop(centroid_blob_data.index[drop_indices])
        # Recalculate dxy and dt after dropping rows
        dxy = centroid_blob_data[["x", "y"]].diff().values[1:]
        dt_raw = np.diff(centroid_blob_data["time"])
    # Convert to milliseconds for speed calculation
    dt = (dt_raw / 1e6).astype(int)  # convert to ms as integers

    # Calculate speed
    centroid_blob_data["speed"] = np.concatenate(([0], np.linalg.norm(dxy, axis=1) / dt / parameters['cm2px'] * 1000))  # cm/s
    k = np.ones(10) / 10  # running avg filter kernel (10 frames)
    centroid_blob_data["speed"] = np.convolve(centroid_blob_data["speed"], k, mode="same")

    # Condition 1: the mice are close to each other
    cond1_frames = np.where(centroid_distances_ffill < parameters['max_distance'])[0]
    # Condition 2: the mean internode distances are within a certain range
    # Condition 2a: the distance between the mice's noses and heads is within a certain range
    cond2a = np.logical_or(nose_head_distances_mouse0 > parameters['max_nose_head_distance'], nose_head_distances_mouse1 > parameters['max_nose_head_distance'])
    # Condition 2b: the mean distance between the mice's own spine nodes is within a certain range
    cond2b = np.logical_or(mean_interspinal_distances_mouse0 > parameters['max_interspinal_distance'], mean_interspinal_distances_mouse1 > parameters['max_interspinal_distance'])
    # Find frames where conditions 2a or 2b are true
    cond2 = np.logical_or(cond2a, cond2b)
    cond2_frames = np.where(cond2)[0]
    # Condition 3: the speed of the blob is above a certain threshold
    cond3_frames = centroid_blob_data[(centroid_blob_data["speed"] > parameters['min_blob_speed'])].index.values

    possible_fights = np.intersect1d(np.intersect1d(cond1_frames, cond2_frames), cond3_frames)
    
    # DIVIDE POSSIBLE FIGHTING FRAMES INTO SUBARRAYS OF CONSECUTIVE FRAMES = ONE POSSIBLE FIGHT
    # Divide possible_tube_test_starts into sub_arrays of consecutive frames (allowing for gaps up to a certain max)
    diffs = np.diff(possible_fights)
    indices = np.where(diffs > parameters['max_frame_gap'])[0]
    indices += 1
    possible_fights = np.split(possible_fights, indices)
    # Filter sub_arrays to keep only those with more than a certain number of frames
    possible_fights = [sub_array for sub_array in possible_fights if len(sub_array) > parameters['min_num_frames']]

    # Include empty frames where the mice were close to each other in the previous frame they were detected
    # If these occur close to or during the time of a possible fight, it's likely the mice are fighting and not detected due to weird poses
    # These frames will have been dropped by condition 2 but can help connect/extend the possible fights detected above
    empty_frames = np.where(np.isnan(tracks).all(axis=(0, 1, 2)))[0]
    empty_frames = np.intersect1d(cond1_frames, empty_frames) # Only select empty frames where the mice were previously close to each other
    possible_fights = np.concatenate(possible_fights)
    possible_fights_w_empty_frames = np.union1d(possible_fights, empty_frames)
    diffs = np.diff(possible_fights_w_empty_frames)
    indices = np.where(diffs > parameters['max_frame_gap_w_empty_frames'])[0]
    indices += 1
    possible_fights_w_empty_frames = np.split(possible_fights_w_empty_frames, indices)
    # Only keep the subarrays that contain at least one frame from the original possible_fights array
    # i.e., don't include subarrays entirely composed of empty frames
    check = [any(frame in possible_fights for frame in sub_array) for sub_array in possible_fights_w_empty_frames]
    possible_fights = [possible_fights_w_empty_frames[i] for i, val in enumerate(check) if val]
    possible_fights = [sub_array for sub_array in possible_fights if len(sub_array) > parameters['min_num_frames']]

    # FILTERING OF POSSIBLE FIGHTS BASED ON THE MEAN INDIVIDUAL SPEEDS OF THE MICE
    fights = []
    # We'll store the end locations for each fight
    fight_end_x = []
    fight_end_y = []
    
    for sub_array in possible_fights:
        start_frame = sub_array[0]-1
        end_frame = sub_array[-1]
        # Clean up identity
        # Trim the centroid data to the frames we are currently considering
        centroid_mouse0_trimmed = centroid_mouse0[:, start_frame:end_frame]
        centroid_mouse1_trimmed = centroid_mouse1[:, start_frame:end_frame]
        # Initialize variables to hold the last known positions of each mouse (used to deal with NaN values in the tracking data)
        last_known_pos0 = centroid_mouse0_trimmed[:, 0]
        last_known_pos1 = centroid_mouse1_trimmed[:, 0]
        # Initialize arrays to hold the cleaned centroid data
        centroid_mouse0_cleaned = centroid_mouse0_trimmed.copy()
        centroid_mouse1_cleaned = centroid_mouse1_trimmed.copy()
        # Loop over the frames from the second frame to the last
        for i in range(1, end_frame-start_frame):
            if np.isnan(centroid_mouse0_trimmed[:, i]).any() and np.isnan(centroid_mouse1_trimmed[:, i]).any():
                continue
            # Calculate the Euclidean distance from each centroid in the current frame to each centroid in the previous frame
            dists = np.zeros((2, 2))
            dists[0, 0] = np.sqrt(np.sum((centroid_mouse0_trimmed[:, i] - last_known_pos0)**2))
            dists[0, 1] = np.sqrt(np.sum((centroid_mouse0_trimmed[:, i] - last_known_pos1)**2))
            dists[1, 0] = np.sqrt(np.sum((centroid_mouse1_trimmed[:, i] - last_known_pos0)**2))
            dists[1, 1] = np.sqrt(np.sum((centroid_mouse1_trimmed[:, i] - last_known_pos1)**2))
            if dists[0, 0] + dists[1, 1] <= dists[0, 1] + dists[1, 0]:
                last_known_pos0 = centroid_mouse0_trimmed[:, i]
                last_known_pos1 = centroid_mouse1_trimmed[:, i] 
            else:
                last_known_pos0 = centroid_mouse1_trimmed[:, i]
                last_known_pos1 = centroid_mouse0_trimmed[:, i]
                centroid_mouse0_cleaned[:, i], centroid_mouse1_cleaned[:, i] = centroid_mouse1_trimmed[:, i].copy(), centroid_mouse0_trimmed[:, i].copy()
        # Calculate centroid speed for each mouse
        mouse0_df = pd.DataFrame(centroid_mouse0_cleaned.T, columns=["x", "y"]).dropna()
        mouse1_df = pd.DataFrame(centroid_mouse1_cleaned.T, columns=["x", "y"]).dropna()
        dt_mouse0 = np.diff(mouse0_df.index.values*1000/fps).astype(int) # ms
        dt_mouse1 = np.diff(mouse1_df.index.values*1000/fps).astype(int) # ms
        dxy_mouse0 = mouse0_df[['x', 'y']].diff().values[1:]
        dxy_mouse1 = mouse1_df[['x', 'y']].diff().values[1:]
        mouse0_df = mouse0_df.iloc[1:]
        mouse1_df = mouse1_df.iloc[1:]
        mouse0_df["speed"] = np.linalg.norm(dxy_mouse0, axis=1) / dt_mouse0 / parameters['cm2px'] * 1000  # cm/s
        mouse1_df["speed"] = np.linalg.norm(dxy_mouse1, axis=1) / dt_mouse1 / parameters['cm2px'] * 1000  # cm/s
        mean_centroid0_speed = mouse0_df["speed"].mean()
        mean_centroid1_speed = mouse1_df["speed"].mean()
        mean_both_centroid_speed = np.nanmean([mean_centroid0_speed, mean_centroid1_speed])
        
        # Add to fights list if either of the mice have a speed above the threshold
        if (mean_centroid0_speed > parameters['min_centroid_speed'] or mean_centroid1_speed > parameters['min_centroid_speed'] or mean_both_centroid_speed > parameters['min_both_centroid_speed']):
            fights.append(sub_array)
            
            # Extract the centroid coordinates for both mice at the last frame
            last_frame_index = end_frame - start_frame - 1
            mouse0_last_pos = centroid_mouse0_cleaned[:, last_frame_index]
            mouse1_last_pos = centroid_mouse1_cleaned[:, last_frame_index]

            # If the last frame has NaN values, search backward to find the most recent valid frame
            valid_frame_found = True
            if np.isnan(mouse0_last_pos).any() or np.isnan(mouse1_last_pos).any():
                # Start from the last frame index in the trimmed array and go backward
                valid_frame_found = False
                search_frame = last_frame_index
                while search_frame >= 0 and not valid_frame_found:  # Note: search from last_frame_index down to 0
                    mouse0_search_pos = centroid_mouse0_cleaned[:, search_frame]
                    mouse1_search_pos = centroid_mouse1_cleaned[:, search_frame]
                    if not np.isnan(mouse0_search_pos).any() and not np.isnan(mouse1_search_pos).any():
                        mouse0_last_pos = mouse0_search_pos
                        mouse1_last_pos = mouse1_search_pos
                        valid_frame_found = True
                        print(f"Found valid frame at trimmed index {search_frame}, original frame {search_frame + start_frame}")
                        print(f"Went back by {last_frame_index - search_frame} frames")
                    search_frame -= 1

            if not valid_frame_found:
                mean_x = np.nan
                mean_y = np.nan
            else:
                # Calculate the mean position of both mice
                mean_x = (mouse0_last_pos[0] + mouse1_last_pos[0]) / 2
                mean_y = (mouse0_last_pos[1] + mouse1_last_pos[1]) / 2

            fight_end_x.append(mean_x)
            fight_end_y.append(mean_y)

    # SAVE FIGHTS
    fight_data = {
        'start_timestamp': [], 
        'end_timestamp': [], 
        'duration (seconds)': [],
        'fight_end_x': [],
        'fight_end_y': []
    }
    
    for idx, subarray in enumerate(fights):
        start_frame = subarray[0]
        end_frame = subarray[-1]
        start_timestamp = frame_timestamps.get(start_frame, 
                              min_time + pd.Timedelta(seconds=start_frame/fps))
        end_timestamp = frame_timestamps.get(end_frame, 
                            min_time + pd.Timedelta(seconds=end_frame/fps))
        duration = (end_timestamp - start_timestamp).total_seconds()
        # Very short fights are likely to be false positives (errors in tracking)
        if duration > 1:
            fight_data['start_timestamp'].append(start_timestamp)
            fight_data['end_timestamp'].append(end_timestamp)
            fight_data['duration (seconds)'].append(duration)
            fight_data['fight_end_x'].append(fight_end_x[idx])
            fight_data['fight_end_y'].append(fight_end_y[idx])

    fights_df = pd.DataFrame(fight_data)

    if video_config['gen_vids'] == True:
        generate_videos(root, fights_df, video_config['camera'], video_config['video_save_path'], video_name_prefix="fight", padding=1)

    return fights_df

def get_experiment_times(
    root: str | os.PathLike, 
    start_time: pd.Timestamp, 
    end_time: pd.Timestamp
) -> DotMap:
    """
    Retrieve experiment start and stop times from environment states
    (i.e. times outside of maintenance mode) occurring within the
    given start and end times.

    Args:
        root (str or os.PathLike): The root path where epoch data is stored.
        start_time (pandas.Timestamp): Start time.
        end_time (pandas.Timestamp): End time.

    Returns:
        DotMap: A DotMap object containing two keys: 'start' and 'stop',
        corresponding to pairs of experiment start and stop times.

    Notes:
    This function uses the last 'Maintenance' event (if available, otherwise
    `end_time`) as the last 'Experiment' stop time. If the first retrieved state
    is 'Maintenance' (e.g. 'Experiment' mode entered before `start_time`),
    `start_time` is used as the first 'Experiment' start time.
    """

    experiment_times = DotMap()
    env_states = api.load(
        root,
        social02.Environment.EnvironmentState,
        start_time,
        end_time,
    )
    if env_states.empty:
        warnings.warn(
            "The environment state df is empty. "
            "Using input `start_time` and `end_time` as experiment times."
        )
        experiment_times.start = [start_time]
        experiment_times.stop = [end_time]
        return experiment_times
    if env_states["state"].iloc[-1] != "Maintenance":
        warnings.warn(
            "No 'Maintenance' event at the end of the search range. "
            "Using input `end_time` as last experiment stop time."
        )
        # Pad with a "Maintenance" event at the end
        env_states = pd.concat(
            [
                env_states,
                pd.DataFrame(
                    "Maintenance",
                    index=[end_time],
                    columns=env_states.columns,
                ),
            ]
        )
    # Use the last "Maintenance" event as end time
    end_time = (env_states[env_states.state == "Maintenance"]).index[-1]
    env_states = env_states[~env_states.index.duplicated(keep="first")]
    # Retain only events between visit start and stop times
    env_states = env_states.iloc[
        env_states.index.get_indexer([start_time], method="bfill")[
            0
        ] : env_states.index.get_indexer([end_time], method="ffill")[0] + 1
    ]
    # Retain only events where state changes (experiment-maintenance pairs)
    env_states = env_states[env_states["state"].ne(env_states["state"].shift())]
    if env_states["state"].iloc[0] == "Maintenance":
        warnings.warn(
            "No 'Experiment' event at the start of the search range. "
            "Using input `end_time` as last experiment stop time."
        )
        # Pad with an "Experiment" event at the start
        env_states = pd.concat(
            [
                pd.DataFrame(
                    "Experiment",
                    index=[start_time],
                    columns=env_states.columns,
                ),
                env_states,
            ]
        )
    experiment_times.start = env_states[
        env_states["state"] == "Experiment"
    ].index.values
    experiment_times.stop = env_states[
        env_states["state"] == "Maintenance"
    ].index.values

    return experiment_times

def exclude_maintenance_data(
    data: pd.DataFrame, 
    experiment_times: DotMap
) -> pd.DataFrame:
    """
    Excludes rows not in experiment times (i.e., corresponding to maintenance times)
    from the given dataframe.

    Args:
        data (pandas.DataFrame): The data to filter. Expected to have a pandas.DateTimeIndex.
        experiment_times (DotMap): A DotMap object containing experiment start and stop times.

    Returns:
        pandas.DataFrame: A pandas DataFrame containing filtered data.
    """
    filtered_data = pd.concat(
        [
            data.loc[start:stop]
            for start, stop in zip(experiment_times.start, experiment_times.stop)
        ]
    )
    return filtered_data

def generate_videos(
    root: str,
    events_df: pd.DataFrame,
    camera: str,
    save_dir: str,
    video_name_prefix: str = "",
    padding: int = 1
) -> None:
    """Generates videos of the events in the events_df DataFrame.

    Args:
        root (str): The root directory of the data.
        events_df (DataFrame): A DataFrame containing the events to generate videos for.
        save_dir (str): The directory to save the videos to.
        video_name_prefix (str, optional): The prefix to use for the video names. It is recommended to use the name of the event type e.g., "tube_test" or "fight".
        padding (int, optional): The number of seconds to add to the start and end of the video.

    Returns:
        None
    """
    root = root.replace("ingest", "raw")
    if video_name_prefix != "" and video_name_prefix[-1] != "_":
        video_name_prefix += "_"
    acquisiton_computer = re.split(r'[\\/]', root)[6]
    experiment = re.split(r'[\\/]', root)[7]
    experiment = re.sub(r'[./\\]', '', experiment) # Remove any slashes or dots
    schemas_module = importlib.import_module(f"aeon.schema.schemas")
    experiment_schema = getattr(schemas_module, experiment, None)
    metadata_reader = getattr(experiment_schema, "Metadata", None)
    metadata = api.load(root, metadata_reader)['metadata'].iloc[0]
    trigger_freq = getattr(getattr(metadata.Devices, camera, None), "TriggerFrequency", None)
    fps = int(getattr(metadata.Devices.VideoController, trigger_freq, None))
    video_reader = getattr(getattr(experiment_schema, camera, None), "Video", None)
    for row in events_df.itertuples():
        start = row.start_timestamp - pd.Timedelta(seconds=padding)
        end = row.end_timestamp + pd.Timedelta(seconds=padding)
        frames_info = api.load(root, video_reader, start, end)
        vid = video.frames(frames_info)
        video_name = f"{video_name_prefix}{acquisiton_computer}_{start.strftime('%Y-%m-%d_%H-%M-%S')}.avi"
        video.export(vid, os.path.join(save_dir, video_name), fps=fps)
    
    return None