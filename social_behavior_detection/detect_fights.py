import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import pandas as pd
import re
from social_behaviour_detection import detect_fights
import time
import json
import sys

def parse_dict(arg_string):
    try:
        return json.loads(arg_string)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError(f"Invalid dictionary format: {arg_string}")

def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--root",
        help="Root directory.",
        required=True,
        type=str
    )
    parser.add_argument(
        "--epoch_time",
        help="Official epoch time for naming the cvs, to be specified in the format YYYY-MM-DDTHH-MM-SS",
        required=False,
        type=str,
        default=None
    )
    parser.add_argument(
        "--start",
        help="Start time of the chunk of interest, to be specified in the format YYYY-MM-DDTHH-MM-SS",
        required=True,
        type=str
    )
    parser.add_argument(
        "--end",
        help="End time of the chunk of interest, to be specified in the format YYYY-MM-DDTHH-MM-SS",
        required=True,
        type=str
    )
    parser.add_argument(
        "--parameters",
        help="The parameters of the detection.",
        required=False,
        type=parse_dict,
        default={
            "cm2px": 5.4,
            "max_distance": 20,
            "max_nose_head_distance": 7,
            "max_interspinal_distance": 10,
            "min_blob_speed": 3,
            "max_frame_gap": 200,
            "min_num_frames": 5,
            "max_frame_gap_w_empty_frames": 100,
            "min_centroid_speed": 20,
            "min_both_centroid_speed": 15
        }
    )
    parser.add_argument(
        "--skeleton", 
        help="A mapping of the required nodes to their corresponding names in your SLEAP project.",
        required=False,
        type=parse_dict,
        default={
            "nose": "nose",
            "head": "head", 
            "right_ear": "right_ear",
            "left_ear": "left_ear",
            "upper_spine": "spine1",
            "centroid": "spine2",
            "lower_spine": "spine3",
            "tail_base": "spine4"
        }
    )
    parser.add_argument(
        "--video_config",
        help="The configuration for generating videos.",
        required=False,
        type=parse_dict,
        default={
            "gen_vids": False,
            "video_save_path": None,
            "camera": "CameraTop"
        }
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory for the csv of fight detection times.",
        required=True,
        type=str
    )
    args = vars(parser.parse_args())

    print("Detecting fights...", flush=True)
    root = args["root"]

    root_split = re.split(r"[\\/]", root) 
    acquisition_computer = root_split[-2]
    exp_name = root_split[-1]
    epoch_time = args["epoch_time"]
    if epoch_time is None:
        time_label = args["start"]
    else:
        time_label = epoch_time
    start = pd.to_datetime(args["start"], format="%Y-%m-%dT%H-%M-%S")
    end = pd.to_datetime(args["end"], format="%Y-%m-%dT%H-%M-%S")

    parameters = args["parameters"]
    skeleton = args["skeleton"]
    video_config = args["video_config"]
    output_dir = args["output_dir"]

    start_time = time.time()
    print(f"Starting fight detection for period: {start} to {end}...")
    fights_df = detect_fights(
        root=root,
        start=start,
        end=end,
        parameters=parameters,
        skeleton=skeleton,
        video_config=video_config
    )

    elapsed_time = time.time() - start_time
    print(f"Detection completed in {elapsed_time:.2f} seconds.")
    print(fights_df)

    # Save to csv
    fights_df.to_csv(f"{output_dir}/fights_{acquisition_computer}_{exp_name}_{time_label}.csv", index=False)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
