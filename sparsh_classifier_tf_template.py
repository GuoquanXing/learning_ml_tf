import math

from IPython import display
# from matplotlib import cm
# from matplotlib import gridspec
# from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

from tf_tmplate import TensorflowUsageSample

tensorflowUsageSampleInstance = TensorflowUsageSample()

# tensorflowUsageSampleInstance.f

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

# tf.estimator.LinearClassifier.

california_housing_dataframe = pd.read_csv("/Users/xingguoquan/Desktop/california_housing_train.csv", sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))

feature_columns = ["latitude",
                   "longitude",
                   "housing_median_age",
                   "total_rooms",
                   "total_bedrooms",
                   "population",
                   "households",
                   "median_income"]
my_label_column = "median_house_value"
my_label_column_new = "median_house_value_is_high"

# Choose the first 12000 (out of 17000) examples for training.
training_examples = tensorflowUsageSampleInstance.preprocess_features(california_housing_dataframe.head(12000),feature_columns)
training_targets = tensorflowUsageSampleInstance.preprocess_targets(california_housing_dataframe.head(12000),my_label_column)

# Choose the last 5000 (out of 17000) examples for validation.
validation_examples = tensorflowUsageSampleInstance.preprocess_features(california_housing_dataframe.tail(5000), feature_columns)
validation_targets = tensorflowUsageSampleInstance.preprocess_targets(california_housing_dataframe.tail(5000), my_label_column)

# Double-check that we've done the right thing.
print("Training examples summary:")
print(display.display(training_examples.describe()))
print("Validation examples summary:")
print(display.display(validation_examples.describe()))

print("Training targets summary:")
display.display(training_targets.describe())
print("Validation targets summary:")
display.display(validation_targets.describe())


linear_classifier = tensorflowUsageSampleInstance.train_linear_classifier_model(
    learning_rate=0.1,
    regularization_strength=0.1,
    steps=300,
    batch_size=100,
    feature_columns=tensorflowUsageSampleInstance.construct_feature_columns(training_examples,feature_columns),
    training_examples=training_examples,
    training_targets=training_targets,
    label_column=my_label_column_new,
    validation_examples=validation_examples,
    validation_targets=validation_targets)
print("Model size:", tensorflowUsageSampleInstance.model_size(linear_classifier))
