# Step 5: SLEAP Model Evaluation
Use the bash script [`sleap_predict.sh`](sleap_predict.sh) and the Jupyter notebook [`composite_video_sleap_labelling_and_evaluating.ipynb`](../step3_SLEAP_labelling/composite_video_sleap_labelling_and_evaluating.ipynb) (from [Step 3](../step3_SLEAP_labelling/README.md)).

## What the code does
`sleap_predict.sh` generates SLEAP predictions on your completely unseen, automatically labelled single animal evaluation SLEAP dataset from Step 3 using your newly trained centroid and centered instance models from Step 4.
`composite_video_sleap_labelling_and_evaluating.ipynb` is used to evaluate the quality of these predictions and the performance of your model.

## Running the code
Activate your SLEAP environment and generate predictions on your unseen evaluation SLEAP dataset from [Step 3](../step3_SLEAP_labelling/README.md) using `sleap_predict.sh`. This script uses the SLEAP CLI command `sleap-track`, documented [here](https://sleap.ai/guides/cli.html#sleap-track). Make sure to use the `only-labelled-frames` flag and you may want to experiment with the `batch_size` number if you are running into HPC resource problems.

Switch back to the `composite_video_sleap_labelling_and_evaluating.ipynb` Jupyter Notebook from Step 3 and activate your SLEAP environment.
1. Load the ground truth and prediction .slp files and evaluate model performance:
	- Rerun the code cell defining the session dictionary and SLEAP imports, then skip to the “evaluate” section
	- Load ground truth and prediction `.slp` files 
	- Compute the model’s score (same metric used for [optimisation of the centered instance model via Optuna](../step4_SLEAP_training/centered_instance_model/README.md), explained [here](../step4_SLEAP_training/centered_instance_model/eval_metric.md))
2. If you're satisfied with the model performance, you should export the model as a frozen graph using SLEAP’s `export_model` function (documented [here](https://sleap.ai/api/sleap.nn.inference.html#sleap.nn.inference.Predictor.export_model)). This is required to use the models in Bonsai and generate predictions over all entire experiments, as described in the [pose and ID Combining Section](../../pose_id_combine/README.md).

## Next Steps
Proceed to the [Pose and ID Combining Section](../../pose_id_combine/README.md).