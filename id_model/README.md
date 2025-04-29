# ID Model

## Overview
This directory contains the complete pipeline for training SLEAP identity models for the Aeon project. Unlike the pose model, which is reusable across experiments, a new ID model needs to be trained whenever new mice are introduced to the Aeon arena. The ID model is trained on quadrant camera videos, which provide better resolution than the top camera, and the predictions are mapped back to the top camera space.

## Workflow
1. **[Calculate Camera Mapping](step1_mapping/README.md)**
   - Create homography matrices between quadrant and top cameras
2. **[Generate Composite Videos](step2_composite_videos/README.md)**
   - Create videos that switch between quadrant cameras to follow mice
3. **[SLEAP Labelling](step3_SLEAP_labelling/README.md)**
   - Create datasets for training and evaluation
4. **[SLEAP Training](step4_SLEAP_training/README.md)**
   - [Train centroid model](step4_SLEAP_training/centroid_model/README.md)
   - [Train centered instance model](step4_SLEAP_training/centered_instance_model/README.md)
5. **[Model Evaluation](step5_SLEAP_evaluation/README.md)**
   - Test models on unseen data

## Next Steps
Proceed to the [Pose and ID Combining Section](../pose_id_combine/README.md).
