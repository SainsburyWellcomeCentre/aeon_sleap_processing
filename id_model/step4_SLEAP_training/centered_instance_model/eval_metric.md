# Evaluation Metric
The evaluation metric is a composite measure that combines identity accuracy and instance detection. The instance detection component is included because relying solely on identity accuracy could lead Optuna to produce low-quality SLEAP models that detect very few mice but happen to identify them correctly by chance, thus artificially boosting their accuracy scores. Identity accuracy is defined as the proportion of correctly identified instances among all detected instances:

$$\text{Identity Accuracy} = \frac{\text{Number of Correctly Identified Instances}}{\text{Total Number of Detected Instances}}$$

Detection performance is quantified using the F1 score:

$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Finally, these two measures are combined via their harmonic mean, which was chosen as it strongly penalizes low values in either metric, ensuring that optimized models perform well in both identification and detection tasks:

$$\text{Combined Score} = 2 \times \frac{\text{Identity Accuracy} \times F_1}{\text{Identity Accuracy} + F_1}$$