# Step 4: SLEAP Model Training

## Overview
In SLEAP, the centroid model is used to find where animals are located in each frame by predicting a single point (the centroid) per instance. It doesn't predict detailed body part positions or directly handle identity assignments—that’s done by the centered instance model. Because of this, the centroid model doesn't need to be as precise or finely tuned as the centered instance model. We typically train our centroid models using manually selected parameter sets that have consistently produced adequate results. However, for the centered instance model, we carefully explore the parameter space using Optuna to achieve optimal performance.

## Training Workflow
1. [Centroid Model Training](centroid_model/README.md)
2. [Centered Instance Model Training](centered_instance_model/README.md)

## Next Steps
Proceed to [Step 5](../step5_SLEAP_evaluation/README.md). 