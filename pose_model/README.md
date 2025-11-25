# Pose Model

## Overview

This directory contains pre-trained SLEAP models for tracking mouse body parts from the top camera view. These models are designed to be reusable across all Aeon experiments, so no training should be necessary for most use cases.

You will also find a [labels](labels) folder containing a `.pkg.slp` versions of the SLEAP labels. These files have embedded frames, so they can be opened in the SLEAP GUI without needing access to the original videos. Some of the `.pkg.slp` files appear as split zip parts (`.z01`, `.z02`, etc.). To use them, simply open the main `.zip` file in your file navigator and click “Extract all”: this will automatically combine and extract the full `.pkg.slp`. 

## Workflow

Simply download the SLEAP centroid model[`AEON_multi_point_topdown_top.centroid`](AEON_multi_point_topdown_top.centroid) and centered instance model [`AEON_multi_point_topdown_top.centered_instance`](AEON_multi_point_topdown_top.centered_instance) and they are ready to use. 

If the models are not performing up to an acceptable standard, it may be because there are variations - such as differences in lighting, arena surroundings, camera angles, etc. - that are not captured in the training data. In this case you should open the SLEAP training labels in the GUI, add some videos the models are struggling on, label and retrain (you can use the same parameters as defined in the models' `initial_config.json` files).

### Notes on the label files and retraining

- The `.pkg.slp` file in the [labels](labels) folder is the one to use if you want to add labels or retrain, as it does not require the original videos. The folder also contains train/val/test splits for reference, but you should use the combined (pre-split) `.pkg.slp` file when adding new labels.  
- The `.slp` files inside the model folders are included only because SLEAP generates them automatically during training. They reference original video paths and therefore cannot be opened in the GUI unless you have those videos. The [labels](labels) folder should contain everything you need.
- If preferred, users can also create their own `.slp` label files from their own videos and fine-tune the provided models on those labels.

## Next Steps
Proceed [ID model Section](../id_model/README.md)
