# Centroid model training
Use the python script [`sleap_train.py`](sleap_train.py) and it’s corresponding bash script [`sleap_train.sh`](sleap_train.sh).

## What the Code Does
The python script trains the centroid model using the labelled datasets created in [Step 3](../../step3_SLEAP_labelling/README.md). 

The process:
1. Loads the labelled `.slp` dataset and splits it into training and validation sets
2. Sets a variety of SLEAP parameters
3. Trains the model

## Running the code
The bash script `sleap_train.sh` calls the python script with the following arguments:
- `file`: Path to `.slp` labels created in [Step 3](../../step3_SLEAP_labelling/README.md)
- `type`: Type of model to train, in this case it should be "centroid"
- `split_labels`: Splits the labels into train/val sets if called. If you train more than one centroid model on the same dataset (e.g., to compare different hyper-parameter combinations), make sure to only use the `--split_labels` flag once so that all your runs train on the same split
- `use_split`: Uses split labels if called. In general this should always be called
- `crop`: Crop size. See [definition](https://sleap.ai/develop/api/sleap.nn.config.data.html#sleap.nn.config.data.InstanceCroppingConfig.crop_size) on the SLEAP website
- `anchor_part`: Text name of a body part (or node) to use as the anchor point. See [definition](https://sleap.ai/develop/api/sleap.nn.config.model.html#sleap.nn.config.model.CentroidsHeadConfig.anchor_part) on the SLEAP website. This should be left as default (“centroid”)
- `input_scale`: Input scaling. See [definition](https://sleap.ai/develop/api/sleap.nn.config.data.html#sleap.nn.config.data.PreprocessingConfig.input_scale) on SLEAP website

Default values should generally be sufficient but it can be helpful to play around with the crop size or input scaling arguments, as well as any of the other variables directly in the code. Activate your SLEAP environment and run the bash script on the HPC with the command `sbatch sleap_train.sh`.

## Next Steps
Proceed to [centered instance model training](../centered_instance_model/README.md).