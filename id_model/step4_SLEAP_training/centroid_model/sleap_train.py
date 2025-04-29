import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import sleap
from sleap.nn.config import *


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--file", help="Path to .slp labels", required=True)
    parser.add_argument(
        "--type",
        help="Type of model to train",
        required=True,
        choices=["centroid", "centered_instance_multiclass"],
    )
    parser.add_argument(
        "--split_labels",
        help="Split labels into train/val/test",
        action="store_true",
    )
    parser.add_argument(
        "--use_split",
        help="Use split labels",
        action="store_true",
    )
    parser.add_argument(
        "--crop",
        help="Crop size",
        default=128,
        type=int,
    )
    parser.add_argument(
        "--anchor_part",
        help="Text name of a body part (node) to use as the anchor point",
        default="centroid",
    )
    parser.add_argument(
        "--input_scale",
        help="Input scaling",
        type=float,
    )
    args = vars(parser.parse_args())

    # set initial parameters
    session_id = Path(args["file"]).stem
    parent_dir = str(Path(args["file"]).parent)
    if args["type"] == "centroid":
        run_name = session_id + "_topdown_top.centroid"
    elif args["type"] == "centered_instance_multiclass":
        run_name = session_id + "_topdown_top.centered_instance_multiclass"
    runs_folder = parent_dir + "/models"
    labels = sleap.load_file(args["file"])

    # initalise default training job config
    cfg = TrainingJobConfig()

    if args["split_labels"]:
        # generate a 0.8/0.1/0.1 train/val/test split
        labels_train, labels_val = labels.split(n=0.9)
        # labels_train, labels_val_test = labels.split(n=0.8)
        # labels_val, labels_test = labels_val_test.split(n=0.5)

        # Save files
        labels_train.save(
            parent_dir + "/" + session_id + ".train.pkg.slp"
        )  # , with_images=True)
        labels_val.save(
            parent_dir + "/" + session_id + ".val.pkg.slp"
        )  # , with_images=True)
        # labels_test.save(
        #     parent_dir + "/" + session_id + ".test.pkg.slp"
        # )  # , with_images=True)

    if args["use_split"]:
        cfg.data.labels.training_labels = (
            parent_dir + "/" + session_id + ".train.pkg.slp"
        )
        cfg.data.labels.validation_labels = (
            parent_dir + "/" + session_id + ".val.pkg.slp"
        )
        # cfg.data.labels.test_labels = parent_dir + "/" + session_id + ".test.pkg.slp"
    else:
        cfg.data.labels.training_labels = args["file"]
        cfg.data.labels.validation_fraction = 0.1

    cfg.data.labels.skeletons = labels.skeletons  # load skeleton

    # preprocessing and training params
    cfg.data.preprocessing.input_scaling = (
        args["input_scale"]
        if args["input_scale"]
        else (0.5 if args["type"] == "centroid" else 1.0)
    )  # 0.75
    cfg.data.instance_cropping.center_on_part = args["anchor_part"]
    # set crop size manually # 128 -top # 544 -nest
    cfg.data.instance_cropping.crop_size = args["crop"]
    cfg.optimization.augmentation_config.rotate = True
    cfg.optimization.epochs = 600  # 200
    cfg.optimization.batch_size = 4  # 8

    cfg.optimization.initial_learning_rate = 0.0001
    cfg.optimization.learning_rate_schedule.reduce_on_plateau = True
    cfg.optimization.learning_rate_schedule.reduction_factor = (
        0.5 if args["type"] == "centroid" else 0.1
    )
    cfg.optimization.learning_rate_schedule.plateau_min_delta = (
        1e-06 if args["type"] == "centroid" else 1e-08
    )
    cfg.optimization.learning_rate_schedule.plateau_patience = 20  # 5
    cfg.optimization.learning_rate_schedule.plateau_cooldown = 3
    cfg.optimization.learning_rate_schedule.min_learning_rate = 1e-08

    cfg.optimization.early_stopping.stop_training_on_plateau = True
    cfg.optimization.early_stopping.plateau_min_delta = 1e-08
    cfg.optimization.early_stopping.plateau_patience = 20  # 10

    # configure nn and model
    cfg.model.backbone.unet = UNetConfig(
        max_stride=16,
        output_stride=2,
        filters=32, # 16
        filters_rate=2.00 if args["type"] == "centroid" else 1.50,
        # up_interpolate=True, # save computations but may lower accuracy
    )
    if args["type"] == "centroid":
        cfg.model.heads.centroid = CentroidsHeadConfig(
            anchor_part=args["anchor_part"], sigma=2.5, output_stride=2
        )
    else:
        confmaps = CenteredInstanceConfmapsHeadConfig(
            anchor_part=args["anchor_part"],
            sigma=1.5,  # 2.5,
            output_stride=2,  # 4,
            loss_weight=1.0,
        )
        # load labels.slp to get track names
        labels = sleap.load_file(args["file"])
        class_vectors = ClassVectorsHeadConfig(
            classes=[track.name for track in labels.tracks],
            output_stride=2,  # 16, #4,
            num_fc_layers=3,
            num_fc_units=256,
            global_pool=True,
            loss_weight=0.01,  # 0.01
        )
        cfg.model.heads.multi_class_topdown = MultiClassTopDownConfig(
            confmaps=confmaps, class_vectors=class_vectors
        )
    # configure outputs
    cfg.outputs.run_name = run_name
    cfg.outputs.save_outputs = True
    cfg.outputs.runs_folder = runs_folder
    cfg.outputs.save_visualizations = True
    cfg.outputs.delete_viz_images = False
    cfg.outputs.checkpointing.initial_model = True
    cfg.outputs.checkpointing.best_model = True

    trainer = sleap.nn.training.Trainer.from_config(cfg)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(e)
