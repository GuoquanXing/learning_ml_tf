import tensorflow as tf

class TensorflowExployer:

    version = tf.__version__
    classifiers = []
    regressors = []
    estimators = []
    estimators = dir(tf.estimator)
    classifiers = []
    regressors = []
    for estimator in estimators:
        if "Classifier" in estimator:
            classifiers.append(estimator)
        if "Regressor" in estimator:
            regressors.append(estimator)
    # Search available optimizers
    dir_train = dir(tf.train)
    optimizers = []
    for optimizer in dir_train:
        if "Optimizer" in optimizer:
            optimizers.append(optimizer)

    def tensorflow_overview(self):
        # Print tensorflow version information
        print("TensorFlow version ", self.version, "Overview:")
        # Print available classifiers
        print("  There are", self.classifiers.__len__(), "available Classifiers: ", self.classifiers)
        # Print available regressors
        print("  There are", self.regressors.__len__(), "available Regressors: ", self.regressors)
        # Print available optimizers
        print("  There are", self.optimizers.__len__(), "available Optimizers: ", self.optimizers)


tf_exployer = TensorflowExployer()
print(tf_exployer.tensorflow_overview())
del tf_exployer
