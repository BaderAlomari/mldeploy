# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Model type: Random Forest Classifier
Hyperparameters: Default parameters used
Author: Bader Alomari
## Intended Use
Whether income exceeds $50K/yr based on census data

## Training Data
Source data: 1994 US Census dataset
Features: 8 categorical features

## Evaluation Data
we split the dataset into a 75/25 train test split to evaluate how the model performs on unseen data

## Metrics
Accuracy: 0.8581
Precision: 0.7363
Recall: 0.6427
F1 Score (F-beta): 0.6864

## Ethical Considerations
income prediction models like this can pick up and produce biases present in the original dataset around race and gender

## Caveats and Recommendations
future improvments could include tuning hyperparameters for better performance