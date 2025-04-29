import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import re
import swc.aeon
from swc.aeon.io import api
from aeon.schema.schemas import social02
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed
import warnings
import os
import shutil
import json
from pathlib import Path
import glob
import sys


def project_to_top_camera(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """
    Transform all points in xy (shape = (N, 2)) using homography H (3x3).
    Returns array of shape (N, 2).
    """
    # Pad with 1s for homogeneous coordinates
    ones = np.ones((xy.shape[0], 1))
    homogeneous = np.hstack((xy, ones))  # shape = (N, 3)
    transformed = homogeneous @ H.T      # shape = (N, 3)

    # Convert back from homogeneous
    transformed_2d = transformed[:, :2] / transformed[:, [2]]
    return transformed_2d
    
def process_minute(minute, timestamps, left_bounds, right_bounds,
                   quad_id_data, top_id_data, top_pose_data, unique_ids, max_distance):
    minute_updates = []  # will store tuples: (pose_ts, skeleton_identifier, matched_identity, matched_likelihood)
    
    # Find indices corresponding to this minute
    ts_minute = timestamps.floor('min')
    indices = np.where(ts_minute == minute)[0]
    
    for i in indices:
        timestamp = timestamps[i]
        lb = left_bounds[i]
        rb = right_bounds[i]
        
        # Process the pose window for this timestamp
        pose_window = top_pose_data.loc[lb:rb].copy()
        if pose_window.empty:
            continue
        unique_ts = pose_window.index.unique()
        if len(unique_ts) > 1:
            warnings.warn(f"Multiple timestamps found within the window {lb}-{rb}. Your FPS could be wrong.")
        pose_ts = unique_ts[0]
        if isinstance(pose_window, pd.Series):
            pose_window = pose_window.to_frame().T
        
        # Filter anchor points from poses
        pose_anchors = pose_window[pose_window['part'].str.contains('anchor')]
        if pose_anchors.empty:
            raise ValueError(f"No anchor points found for timestamp {timestamp}.")
        pose_coords = pose_anchors[['x', 'y']].values.astype(float)
        
        # Get ID data: try quad data first, then top ID data
        quad_window = quad_id_data.loc[lb:rb].copy()
        if quad_window.empty:
            id_window = top_id_data.loc[lb:rb].copy()
        else:
            id_window = quad_window
        if id_window.empty:
            continue
        if isinstance(id_window, pd.Series):
            id_window = id_window.to_frame().T

        # Use projected coordinates if available
        if 'x_top' in id_window.columns:
            id_coords = id_window[['x_top', 'y_top']].values.astype(float)
        else:
            id_coords = id_window[['x', 'y']].values.astype(float)
        
        unique_ids_window = id_window['identity'].unique().tolist()
        cost_matrix = np.full((len(pose_coords), len(unique_ids_window)), np.inf)
        candidate_matrix = np.empty((len(pose_coords), len(unique_ids_window)), dtype=object)
        
        # Process each candidate in the id_window
        for j in range(len(id_coords)):
            dists = np.sqrt(np.sum((pose_coords - id_coords[j])**2, axis=1))
            if dists.min() > max_distance:
                # warnings.warn(f"Could not match ID {id_window.iloc[j]['identity']} detected in camera {id_window.iloc[j]['camera']} at timestamp {timestamp} to any pose.")
                pass
            else:
                row_idx = dists.argmin()
                col_idx = unique_ids_window.index(id_window.iloc[j]["identity"])
                likelihood = id_window.iloc[j]["identity_likelihood"][id_window.iloc[j]["identity"]]
                cost_val = -likelihood  # higher likelihood gives lower cost
                if cost_val < cost_matrix[row_idx, col_idx]:
                    cost_matrix[row_idx, col_idx] = cost_val
                    candidate_matrix[row_idx, col_idx] = id_window.iloc[j]
        
        # Check if the entire cost matrix is infeasible
        if np.all(np.isinf(cost_matrix)):
            # warnings.warn(f"No feasible pose ID matches found for timestamp {timestamp}.")
            continue

        # Identify rows (pose anchors) that have at least one feasible candidate
        valid_rows = np.where(~np.all(np.isinf(cost_matrix), axis=1))[0]
        if valid_rows.size < cost_matrix.shape[0]:
            # warnings.warn(f"Some poses at timestamp {timestamp} have no feasible ID match; they will be skipped.")
            pass

        # Identify columns (candidate IDs) that have at least one feasible pose
        valid_cols = np.where(~np.all(np.isinf(cost_matrix), axis=0))[0]
        if valid_cols.size < cost_matrix.shape[1]:
            # warnings.warn(f"Some IDs at timestamp {timestamp} have no feasible pose match; they will be skipped.")
            pass

        # Create a reduced cost matrix with only valid rows
        reduced_cost_matrix = cost_matrix[np.ix_(valid_rows, valid_cols)]
        row_ind_reduced, col_ind_reduced = linear_sum_assignment(reduced_cost_matrix)
        # Map back to the original cost matrix indices
        row_ind = valid_rows[row_ind_reduced]
        col_ind = valid_cols[col_ind_reduced]

        # Collect the valid assignments as updates
        assigned_ids = set()
        assigned_poses = set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < np.inf:
                pose_row = pose_anchors.iloc[r]
                id_row = candidate_matrix[r, c]
                assigned_ids.add(id_row['identity'])
                assigned_poses.add(pose_row['identity'])
                minute_updates.append((pose_ts, pose_row['identity'], id_row['identity'], id_row['identity_likelihood']))
        
        # If exactly one ID is missing and one pose is unassigned, infer the missing assignment
        missing_ids = set(unique_ids) - assigned_ids
        unassigned_poses = set(pose_anchors['identity']) - assigned_poses
        if len(missing_ids) == 1 and len(unassigned_poses) == 1:
            missing_id = list(missing_ids)[0]
            minute_updates.append((pose_ts, list(unassigned_poses)[0], missing_id, {uid: np.NaN for uid in unique_ids}))
    
    return minute_updates

def match_ids_to_poses_parallel_per_minute(quad_id_data, top_id_data, top_pose_data,
                                             timestamps, left_bounds, right_bounds,
                                             max_distance=40, n_jobs=-1):
    result_data = top_pose_data.copy()
    unique_ids = top_id_data['identity'].unique().tolist()
    
    # Group timestamps by minute (floor to minute)
    timestamps_minute = timestamps.floor('min')
    unique_minutes = np.unique(timestamps_minute)
    
    # Process each minute in parallel
    all_updates = Parallel(n_jobs=n_jobs)(
        delayed(process_minute)(
            minute, timestamps, left_bounds, right_bounds,
            quad_id_data, top_id_data, top_pose_data, unique_ids, max_distance
        ) for minute in unique_minutes
    )
    
    # Batch all updates into a DataFrame
    updates_list = [
        {'time': pose_ts, 'skeleton_identifier': skeleton_identifier, 
        'matched_identity': matched_identity, 'matched_identity_likelihood': matched_likelihood}
        for updates in all_updates
        for pose_ts, skeleton_identifier, matched_identity, matched_likelihood in updates
    ]
    
    if updates_list:
        updates_df = pd.DataFrame(updates_list)
    else:
        print("No updates to apply.")
        return result_data

    # Convert timestamp column to datetime if not already
    updates_df['time'] = pd.to_datetime(updates_df['time'])

    # Reset the index on the result_data so timestamp becomes a column
    result_data.reset_index(inplace=True)
    result_data.rename(columns={'index': 'time'}, inplace=True)
    
    # Merge on timestamp and skeleton_identifier to bring in the new updates
    merged = pd.merge(result_data, updates_df, how='left',
                      left_on=['time', 'identity'],
                      right_on=['time', 'skeleton_identifier'])
    
    # Where an update exists, overwrite the identity and identity_likelihood
    mask = merged['matched_identity'].notna()
    merged.loc[mask, 'identity'] = merged.loc[mask, 'matched_identity']
    merged.loc[mask, 'identity_likelihood'] = merged.loc[mask, 'matched_identity_likelihood']
    
    # Clean up the merged DataFrame and set the index back to timestamp
    merged = merged.drop(columns=['skeleton_identifier', 'matched_identity', 'matched_identity_likelihood'])
    merged = merged.set_index('time')
    
    return merged


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--root",
        help="Root directory.",
        required=True,
        type=str
    )
    parser.add_argument(
        "--start",
        help="Start time of the chunk of interest, to be specified in the format YYYY-MM-DDTHH-MM-SS",
        required=True,
        type=str
    )
    parser.add_argument(
        "--fps",
        help="Frames per second used for computing end time offset (default: 50).",
        required=False,
        type=float,
        default=50
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory for the processed data",
        required=True,
        type=str
    )
    parser.add_argument(
        "--job_id",
        help="Job ID to be used for the matched pose ID data config file",
        required=True,
        type=int,
    )

    args = vars(parser.parse_args())

    root = args["root"]
    fps = args["fps"]
    start = pd.Timestamp(args["start"]).tz_localize(None)
    tolerance = pd.Timedelta(seconds=(1/fps) / 4)
    end = start + pd.Timedelta(hours=1) - pd.Timedelta(seconds=1/fps-tolerance.total_seconds())
    output_dir = args["output_dir"]
    job_id = args["job_id"]

    output_file = os.path.join(output_dir, f"CameraTop_222_{job_id}_{args['start']}.bin")
    pattern = os.path.join(output_dir, f"CameraTop_222_*_{args['start']}.bin")
    matching_files = glob.glob(pattern)
    if matching_files:
        print(f"Matching output file(s) already exist: {matching_files}, skipping.")
        return

    parts = re.split(r"[\\/]", root)  # Splits on either slash
    acquisition_computer = parts[-2]
    exp_name = parts[-1]
    quad_cameras = ["CameraNorth", "CameraSouth", "CameraEast", "CameraWest"]
    homography_paths = [f'/ceph/aeon/aeon/code/scratchpad/Orsi/pixel_mapping/pixel_mapping_results/{exp_name}/{acquisition_computer}/H_{camera}.npy'
                        for camera in quad_cameras]
    homographies = [np.load(path) for path in homography_paths]

    # Load the data
    quad_id_data = []
    for camera in quad_cameras:
        r = swc.aeon.io.reader.Pose(pattern=f"{camera}_202*")
        df = api.load(root, r, start=start, end=end) 
        if not df.empty:
            df["camera"] = camera
            H = homographies[quad_cameras.index(camera)]
            xy = df[["x", "y"]].values
            xy_projected = project_to_top_camera(H, xy)
            df["x_top"] = xy_projected[:, 0]
            df["y_top"] = xy_projected[:, 1]
            quad_id_data.append(df)
    quad_id_data = pd.concat(quad_id_data).sort_index()
    quad_id_data = quad_id_data[quad_id_data["part"].str.contains("anchor")]
    
    r = swc.aeon.io.reader.Pose(pattern=f"CameraTop_202*")
    top_id_data = api.load(root, r, start=start, end=end) 
    top_id_data["camera"] = "CameraTop"
    top_id_data = top_id_data[top_id_data["part"].str.contains("anchor")]

    r = swc.aeon.io.reader.Pose(pattern=f"CameraTop_212*")
    top_pose_data = api.load(root, r, start=start, end=end)

    if top_pose_data.empty or quad_id_data.empty:
        print(f"No data found for this chunk. Saving empty file.")
        with open(output_file, 'wb') as f:
            pass
        return

    # Create new config file to pass to HARP bin file writer
    pose_id_config_dir = f"/ceph/aeon/aeon/data/processed/222/{job_id}"
    if not os.path.exists(pose_id_config_dir):
        print(f"Creating pose ID config file {pose_id_config_dir}/confmap_config.json")
        os.makedirs(pose_id_config_dir)
        pose_config_path = f"/ceph/aeon/aeon/data/processed/{top_pose_data['model'][0]}/confmap_config.json"
        id_config_path = f"/ceph/aeon/aeon/data/processed/{quad_id_data['model'][1]}/confmap_config.json"
        shutil.copy(id_config_path, pose_id_config_dir)
        with open(pose_config_path) as f:
            pose_data = json.load(f)
        with open(f'{pose_id_config_dir}/confmap_config.json') as f:
            combined_data = json.load(f)
        combined_data['model']['heads']['multi_class_topdown']['confmaps']['anchor_part'] = pose_data['model']['heads']['centered_instance']['anchor_part']
        combined_data['model']['heads']['multi_class_topdown']['confmaps']['part_names'] = pose_data['model']['heads']['centered_instance']['part_names']
        with open(f'{pose_id_config_dir}/confmap_config.json', 'w') as f:
            json.dump(combined_data, f, indent=4)
    else:  
        print(f"Full pose ID config file already exists: {pose_id_config_dir}")

    timestamps = pd.date_range(start=start, end=end, freq=pd.DateOffset(seconds=1/fps))
    timestamps_array = timestamps.values

    # Precompute the bounds for all timestamps
    left_bounds = timestamps_array - tolerance
    right_bounds = timestamps_array + tolerance

    pose_id_data = match_ids_to_poses_parallel_per_minute(quad_id_data, top_id_data, top_pose_data,
                                                          timestamps, left_bounds, right_bounds)
    # Drop poses that were not matched to any ID
    pose_id_data = pose_id_data[~pose_id_data["identity"].apply(lambda x: isinstance(x, float))]

    # Save the data to a binary file
    # output_file_csv = os.path.join(output_dir, f"CameraTop_222_{job_id}_{args['start']}.csv")
    # pose_id_data.to_csv(output_file_csv)
    pose_id_data.index = api.to_seconds(pose_id_data.index)
    r = swc.aeon.io.reader.Pose(pattern=f"CameraTop_222*")
    r.write(file = Path(output_file), data = pose_id_data)
    # Check if the file was created and save an empty file if not
    if not os.path.exists(output_file):
        print(f"Output file {output_file} was not created. Saving an empty file.")
        with open(output_file, 'wb') as f:
            pass


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
