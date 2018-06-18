import pandas as pd
import tensorflow as tf
import numpy as np

# Load data in csv format
dataset_training = pd.read_csv('/Users/xingguoquan/Documents/workspace/kaggle/Titanic/train.csv')
dataset_testing = pd.read_csv('/Users/xingguoquan/Documents/workspace/kaggle/Titanic/test.csv')
dataset_testing_submit = pd.read_csv('/Users/xingguoquan/Documents/workspace/kaggle/Titanic/gender_submission.csv')

# print(dataset_training.describe())
# print(dataset_training.head(20))
# print(dataset_training.keys())
dataset_training['Sex'] = dataset_training['Sex'].apply(lambda x: 1 if x == 'male' else 0)
dataset_testing['Sex'] = dataset_testing['Sex'].apply(lambda x: 1 if x == 'male' else 0)


# construct features set and label set
def preprocess_dataset(name_columns, dataset):
    features = pd.DataFrame()
    for feature in name_columns:
        features[feature] = dataset[feature]
    return features


features = ['Pclass', 'Sex', 'Age']
# features = ['Pclass','Sex','Age','Ticket','SibSp','Fare','Cabin','Embarked']
# build train data set
dataset_training_feature = preprocess_dataset(features, dataset_training)
dataset_training_target = preprocess_dataset(['Survived'], dataset_training)
# build test data set
dataset_testing_feature = preprocess_dataset(features, dataset_testing)
dataset_testing_target = preprocess_dataset(['Survived'], dataset_testing_submit)

dataset_training_feature = dataset_training_feature.fillna(0)
dataset_training_feature = dataset_training_feature.fillna(0)
dataset_testing_feature = dataset_testing_feature.fillna(0)

# construct feature columns to tell model the feature metadata
def construct_feature_columns(dataset_training_feature):
    # Create feature columns for all features.
    my_feature_columns = []
    for key in dataset_training_feature.keys():
        if key == 'Embarked':
            '''
            # print("hello, line 44")
            # my_feature_columns.append(
            #     tf.feature_column.categorical_column_with_vocabulary_list(
            #     key=key,
            #     vocabulary_list=["C", "Q", "S"])
            # )
            '''
            pass
        else:
            my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    return my_feature_columns


# instance a model
classifier = tf.estimator.DNNClassifier(
        feature_columns=construct_feature_columns(dataset_training_feature),
        hidden_units=[10, 10])


# def train input function for model to train
def train_input_fn(features, labels, batch_size):
    features_new = {key: np.array(value) for key, value in dict(features).items()}
    dataset = tf.data.Dataset.from_tensor_slices((features_new, labels))
    dataset = dataset.shuffle(buffer_size=1000).repeat(count=None).batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


def eval_input_fn(features, labels=None, batch_size=None):
    """An input function for evaluation or prediction"""
    features_new = {key: np.array(value) for key, value in dict(features).items()}
    features = features_new
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert inputs to a tf.dataset object.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


train_input_fn(dataset_training_feature, dataset_training_target, 10)

# train the model
classifier.train(
        input_fn=lambda:train_input_fn(dataset_training_feature, dataset_training_target, 10), steps=300)

# Evaluate the model.
eval_result = classifier.evaluate(
    input_fn=lambda:eval_input_fn(dataset_testing_feature, dataset_testing_target, batch_size=1))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
