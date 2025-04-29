# Pose Model

## Overview

This directory contains pre-trained SLEAP models for tracking mouse body parts from the top camera view. These models are designed to be reusable across all Aeon experiments, so no training should be necessary for most use cases.

## Workflow

Simply download the SLEAP centroid model[ `AEON_multi_point_topdown_top.centroid`](AEON_multi_point_topdown_top.centroid) and centered instance model [`AEON_multi_point_topdown_top.centered_instance`](AEON_multi_point_topdown_top.centered_instance) and they are ready to use. 

If the models are not performing up to an acceptable standard, it may be because there are variations - such as differences in lighting, arena surroundings, camera angles, etc. - that are not captured in the training data. In this case you should open the SLEAP training labels in the GUI, add some videos the models are struggling on, label and retrain (you can use the same parameters as defined in the models' `initial_config.json` files).

## Next Steps
Proceed [ID model Section](../id_model/README.md)
