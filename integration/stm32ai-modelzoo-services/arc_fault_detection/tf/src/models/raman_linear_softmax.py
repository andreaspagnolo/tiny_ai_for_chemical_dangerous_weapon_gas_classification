"""Train-from-scratch linear Raman classifier for Model Zoo Services 4.1.1."""

from tensorflow import keras
from tensorflow.keras import layers


def get_raman_linear_softmax(input_shape=(1, 512, 1), num_classes=3):
    """Return a fully quantizable Flatten-Dense-Softmax network.

    Model Zoo Services initializes and trains these weights normally. No
    coefficient from the scikit-learn logistic-regression baseline is loaded.
    """
    inputs = keras.Input(shape=input_shape, name="raman_spectrum")
    x = layers.Flatten(name="flatten")(inputs)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_initializer="glorot_uniform",
        kernel_regularizer=keras.regularizers.l2(1.0e-4),
        name="class_probabilities",
    )(x)
    return keras.Model(inputs, outputs, name="raman_linear_softmax")
