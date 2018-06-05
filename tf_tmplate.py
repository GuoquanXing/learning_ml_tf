import math

from IPython import display
# from matplotlib import cm
# from matplotlib import gridspec
# from matplotlib import pyplot as plt`
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset


class TensorflowUsageSample:

    def __init__(self):
        print("Constructor called")

    def function1(self):
        print("function 1 called")

    def instance_func(self, name):
        print(name)

    def my_input_fn(self, features, targets, batch_size=1, shuffle=True, num_epochs=None):
        """Trains a linear regression model.

      Args:
        features: pandas DataFrame of features
        targets: pandas DataFrame of targets
        batch_size: Size of batches to be passed to the model
        shuffle: True or False. Whether to shuffle the data.
        num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
      Returns:
        Tuple of (features, labels) for next data batch
      """
        # Convert pandas data into a dict of np arrays.
        features = {key: np.array(value) for key, value in dict(features).items()}

        # Construct a dataset, and configure batching/repeating.
        ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
        ds = ds.batch(batch_size).repeat(num_epochs)

        # Shuffle the data, if specified.
        if shuffle:
            ds = ds.shuffle(10000)

        # Return the next batch of data.
        features, labels = ds.make_one_shot_iterator().get_next()
        return features, labels

    def preprocess_features(self, california_housing_dataframe, feature_columns):
        """Prepares input features from California housing data set.

        Args:
          california_housing_dataframe: A Pandas DataFrame expected to contain data
            from the California housing data set.
        Returns:
          A DataFrame that contains the features to be used for the model, including
          synthetic features.
        """
        # selected_features = california_housing_dataframe[
        #     ["latitude",
        #      "longitude",
        #      "housing_median_age",
        #      "total_rooms",
        #      "total_bedrooms",
        #      "population",
        #      "households",
        #      "median_income"]]
        selected_features = california_housing_dataframe[feature_columns]
        processed_features = selected_features.copy()
        # Create a synthetic feature.
        processed_features["rooms_per_person"] = (
                california_housing_dataframe["total_rooms"] /
                california_housing_dataframe["population"])
        return processed_features

    def preprocess_targets(self, california_housing_dataframe, label_column):
        """Prepares target features (i.e., labels) from California housing data set.

        Args:
          california_housing_dataframe: A Pandas DataFrame expected to contain data
            from the California housing data set.
        Returns:
          A DataFrame that contains the target feature.
        """
        output_targets = pd.DataFrame()
        # Create a boolean categorical feature representing whether the
        # median_house_value is above a set threshold.
        output_targets["median_house_value_is_high"] = (
                california_housing_dataframe[label_column] > 265000).astype(float)
        return output_targets

    def get_quantile_based_buckets(self, feature_values, num_buckets):
        quantiles = feature_values.quantile(
            [(i + 1.) / (num_buckets + 1.) for i in range(num_buckets)])
        return [quantiles[q] for q in quantiles.keys()]

    def construct_feature_columns(self, training_examples, feature_columns):
        """Construct the TensorFlow Feature Columns.

        Returns:
          A set of feature columns`
        """
        bucketized_features = []
        for feature in feature_columns:
            bucketized_features.append(tf.feature_column.bucketized_column(
                tf.feature_column.numeric_column(feature),
                boundaries=self.get_quantile_based_buckets(training_examples[feature], 10))
            )

        # bucketized_households = tf.feature_column.bucketized_column(
        #     tf.feature_column.numeric_column("households"),
        #     boundaries=get_quantile_based_buckets(training_examples["households"], 10))

        # long_x_lat = tf.feature_column.crossed_column(
        #     set([bucketized_longitude, bucketized_latitude]), hash_bucket_size=1000)

        feature_columns_buckrtized = set()
        for bucketized_data in bucketized_features:
            feature_columns_buckrtized.add(bucketized_data)
        # feature_columns = set([
        #     long_x_lat,
        #     bucketized_longitude,
        #     bucketized_latitude,
        #     bucketized_housing_median_age,
        #     bucketized_total_rooms,
        #     bucketized_total_bedrooms,
        #     bucketized_population,
        #     bucketized_households,
        #     bucketized_median_income,
        #     bucketized_rooms_per_person])

        return feature_columns_buckrtized

    def model_size(self, estimator):
        variables = estimator.get_variable_names()
        size = 0
        for variable in variables:
            if not any(x in variable
                       for x in ['global_step',
                                 'centered_bias_weight',
                                 'bias_weight',
                                 'Ftrl']
                       ):
                size += np.count_nonzero(estimator.get_variable_value(variable))
        return size

    def train_linear_classifier_model(
            self, learning_rate,
            regularization_strength,
            steps,
            batch_size,
            feature_columns,
            training_examples,
            training_targets,
            label_column,
            validation_examples,
            validation_targets,
            my_optimizer='FtrlOptimizer'):
        """Trains a linear regression model.

        In addition to training, this function also prints training progress information,
        as well as a plot of the training and validation loss over time.

        Args:
          learning_rate: A `float`, the learning rate.
          regularization_strength: A `float` that indicates the strength of the L1
             regularization. A value of `0.0` means no regularization.
          steps: A non-zero `int`, the total number of training steps. A training step
            consists of a forward and backward pass using a single batch.
          feature_columns: A `set` specifying the input feature columns to use.
          training_examples: A `DataFrame` containing one or more columns from
            `california_housing_dataframe` to use as input features for training.
          training_targets: A `DataFrame` containing exactly one column from
            `california_housing_dataframe` to use as target for training.
          label_column: The column name for label
          validation_examples: A `DataFrame` containing one or more columns from
            `california_housing_dataframe` to use as input features for validation.
          validation_targets: A `DataFrame` containing exactly one column from
            `california_housing_dataframe` to use as target for validation.
          my_optimizer: (Optional)the optimizer given for GD implementation

        Returns:
          A `LinearClassifier` object trained on the training data.
        """

        periods = 7
        steps_per_period = steps / periods

        if my_optimizer == 'FtrlOptimizer':
            my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate,
                                                  l1_regularization_strength=regularization_strength)
            my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
        else:
            my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            # my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

        # Create a linear classifier object.
        linear_classifier = tf.estimator.LinearClassifier(
            feature_columns=feature_columns,
            optimizer=my_optimizer
        )

        # Create input functions.
        training_input_fn = lambda: self.my_input_fn(training_examples,
                                                     training_targets[label_column],
                                                     batch_size=batch_size)
        predict_training_input_fn = lambda: self.my_input_fn(training_examples,
                                                             training_targets[label_column],
                                                             num_epochs=1,
                                                             shuffle=False)
        predict_validation_input_fn = lambda: self.my_input_fn(validation_examples,
                                                               validation_targets[label_column],
                                                               num_epochs=1,
                                                               shuffle=False)

        # Train the model, but do so inside a loop so that we can periodically assess
        # loss metrics.
        print
        "Training model..."
        print
        "LogLoss (on validation data):"
        training_log_losses = []
        validation_log_losses = []
        for period in range(0, periods):
            # Train the model, starting from the prior state.
            linear_classifier.train(
                input_fn=training_input_fn,
                steps=steps_per_period
            )
            # Take a break and compute predictions.
            training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)
            training_probabilities = np.array([item['probabilities'] for item in training_probabilities])

            validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
            validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])

            # Compute training and validation loss.
            training_log_loss = metrics.log_loss(training_targets, training_probabilities)
            validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
            # Occasionally print the current loss.
            print(" period {0:02d} : training loss - {1:0.2f}, validation loss - {2:0.2f}".format(
                period, validation_log_loss, validation_log_loss))
            # Add the loss metrics from this period to our list.
            training_log_losses.append(training_log_loss)
            validation_log_losses.append(validation_log_loss)
        print("Model training finished.")

        # Output a graph of loss metrics over periods.
        # plt.ylabel("LogLoss")
        # plt.xlabel("Periods")
        # plt.title("LogLoss vs. Periods")
        # plt.tight_layout()
        # plt.plot(training_log_losses, label="training")
        # plt.plot(validation_log_losses, label="validation")
        # plt.legend()

        return linear_classifier
