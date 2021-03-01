import numpy as np
import tensorflow as tf

from scipy.stats import pearsonr
from typing import Tuple, List, Callable

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.math import count_nonzero
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Input,
    Lambda,
    Layer,
    LeakyReLU,
    TimeDistributed,
)

from spektral.data import BatchLoader
from spektral.layers import ECCConv, GlobalAttnSumPool

from gcnn.datasets import GraphDB
from gcnn.metrics import pearson, r_squared


def train_model(
    dataset: GraphDB,
    tf_loss: Callable,
    metrics: List[Callable],
    channels: List[int],
    n_layers: int = 1,
    batch_size: int = 32,
    number_epochs: int = 40,
    learning_rate: float = 0.001,
    summary: bool = False,
    verbose: bool = False,
) -> Tuple:
    """Comment

    Args:
        dataset: GraphDB instance
        tf_loss: tensorflow-type loss function
        metrics: list of metric functions
        n_layers: number of ECCConv layers in model
        channels: number of convolutional channels per layer
        batch_size: size of mini-batches
        number_epochs: number of epochs
        learning_rate: optimizer learning rate
        summary: print model summary
        verbose: print training data

    Returns:
        trained model and training history

    """
    # Dimension of node features
    size_nodes = dataset.n_node_features
    # Dimension of edge features
    size_feats = dataset.n_edge_features

    # Create GCN model
    model = create_gcnn(
        nodes_shape=size_nodes,
        edges_shape=size_feats,
        channels=channels,
        n_layers=n_layers,
    )

    # Compile GCN
    model.compile(
        # Adam optimizer that handles sparse
        # updates more efficiently
        optimizer=Adam(learning_rate),
        # List of metrics to monitor
        metrics=metrics,
        # Objective function
        loss=tf_loss,
    )

    # Run model in Eager mode
    model.run_eagerly = True

    # Summary
    if summary:
        model.summary()

    # Loader returns batches of graphs
    # with zero-padding done batch-wise
    loader = BatchLoader(dataset, batch_size=batch_size, shuffle=True)

    # Trains the model
    history = model.fit(
        loader.load(),
        # Show training info
        verbose=verbose,
        # training cycles
        epochs=number_epochs,
        # len(dataset) // batch_size
        steps_per_epoch=loader.steps_per_epoch,
    )

    return model, history


def evaluate_model(model, tests_set):

    # Loader returns batches of graphs
    # with zero-padding done batch-wise
    loader = BatchLoader(tests_set, batch_size=1, shuffle=False)

    # generate a pair (predicted affinity & sigma) per graph
    prediction = model.predict(loader.load(), steps=len(tests_set))

    # discarding sigma
    pred_values = prediction[::2]
    
    # experimental affinity values &
    # censured data indexes
    true_values = np.array([
        tests_set[i]["y"][2] for i in range(tests_set.n_graphs)])
    lefts_indexes = np.array([
        tests_set[i]["y"][0] for i in range(tests_set.n_graphs)])
    right_indexes = np.array([
        tests_set[i]["y"][1] for i in range(tests_set.n_graphs)])

    # non-censored data indexes
    inner_indexes = (1 - right_indexes) * (1 - lefts_indexes)

    ##################################
    # Performance metrics
    ##################################

    metrics = {
        "mae": 0,
        "mse": 0,
        "pearson": 0,
        "fraction_lefts_outliers": 0,
        "fraction_right_outliers": 0,
    }

    true = true_values[(inner_indexes > 0)].flatten()
    pred = pred_values[(inner_indexes > 0)].flatten()
    
    # estimate MEAN ABSOLUTE ERROR
    mae = tf.keras.losses.MeanAbsoluteError()
    metrics["mae"] = mae(true, pred).numpy()

    # estimate MEAN SQUARED ERROR
    mae = tf.keras.losses.MeanSquaredError()
    metrics["mse"] = mae(true, pred).numpy()

    # estimate PEARSON CORRELATION
    metrics["pearson"] = pearsonr(true, pred)[0]

    # Number of predicted values
    # above or below the correct threashold
    lefts_outliers = tf.nn.relu(
        pred_values[(lefts_indexes > 0)] - true_values[(lefts_indexes > 0)])
    right_outliers = tf.nn.relu(
        true_values[(right_indexes > 0)] - pred_values[(right_indexes > 0)])

    # estimate FRACTION of censored values
    # predicted higher than true boundaries
    metrics["fraction_lefts_outliers"] = count_nonzero(lefts_outliers) / sum(lefts_indexes)
    metrics["fraction_right_outliers"] = count_nonzero(right_outliers) / sum(right_indexes)

    return metrics


def create_gcnn(nodes_shape, edges_shape, channels, n_layers):

    X = Input(shape=(None, nodes_shape))
    A = Input(shape=(None, None))
    E = Input(shape=(None, None, edges_shape))

    x = Lambda(lambda x: tf.cast(x, tf.float32))(X)
    a = Lambda(lambda x: tf.cast(x, tf.float32))(A)
    e = Lambda(lambda x: tf.cast(x, tf.float32))(E)

    f = TimeDistributed(Dense(8, use_bias=False))(e)

    for i in range(n_layers):
        x = ECCConv(channels[i], activation="tanh")([x, a, f])
        x = BatchNormalization()(x)

    x = GlobalAttnSumPool()(x)
    x = Dense(1024, LeakyReLU(alpha=0.01))(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization()(x)

    output = MLEDense(1)(x)
    return Model(inputs=[X, A, E], outputs=output)


class MLEDense(Layer):
    def __init__(self, units=1):
        super(MLEDense, self).__init__()
        self.units = units

    def build(self, input_shape):

        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            name='final_weight',
            trainable=True,
        )

        self.b = self.add_weight(
            shape=(self.units,), 
            initializer="random_normal", 
            name='final_bias',
            trainable=True)

        # pseudo-prior of variance
        init = tf.random_normal_initializer(mean=0.0, stddev=1.0)

        # variance of error distribution
        self.sigma = tf.Variable(
            init(shape=(self.units, self.units)), 
            name='sigma', 
            trainable=True)

    def call(self, inputs):
        y = tf.matmul(inputs, self.w) + self.b
        return tf.concat([y, self.sigma], axis=0)
    
    def get_config(self):
        config = super(MLEDense, self).get_config()
        config.update({'units': self.units})
        return config
