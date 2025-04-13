from typing import Tuple, List, Callable

import numpy as np
import tensorflow as tf

from scipy.stats import pearsonr
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
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


@tf.autograph.experimental.do_not_convert
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
    """Train a GCN model.

    Args:
        dataset: dataset of graphs
        tf_loss: tensorflow loss function
        metrics: list of tensorflow metrics
        channels: number of channels per layer
        n_layers: number of layers
        batch_size: batch size
        number_epochs: number of epochs
        learning_rate: learning rate
        summary: show model summary
        verbose: show training info

    Returns:
        model: trained GCN model
        history: training history

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


def evaluate_model(model: Model, tests_set: GraphDB) -> dict:
    """Evaluate a GCN model.

    Args:
        model: trained GCN model
        tests_set: dataset of graphs

    Returns:
        metrics: dictionary of performance metrics

    """
    # Loader returns batches of graphs
    # with zero-padding done batch-wise
    loader = BatchLoader(tests_set, batch_size=1, shuffle=False)

    # generate a pair (predicted affinity & sigma) per graph
    prediction = model.predict(loader.load(), steps=len(tests_set))

    # discarding sigma
    pred_values = prediction[::2]

    # experimental affinity values
    true_values = np.array(
        [tests_set[i]["y"][2] for i in range(tests_set.n_graphs)])

    # censored data indexes (left & right) boundaries
    lefts_indexes = np.array(
        [tests_set[i]["y"][0] for i in range(tests_set.n_graphs)])
    right_indexes = np.array(
        [tests_set[i]["y"][1] for i in range(tests_set.n_graphs)])

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
    lefts_outliers = tf.nn.relu(pred_values[(lefts_indexes > 0)] -
                                true_values[(lefts_indexes > 0)])
    right_outliers = tf.nn.relu(true_values[(right_indexes > 0)] -
                                pred_values[(right_indexes > 0)])

    # estimate FRACTION of censored values
    # predicted higher than true boundaries
    metrics["fraction_lefts_outliers"] = tf.math.count_nonzero(
        lefts_outliers) / sum(lefts_indexes)
    metrics["fraction_right_outliers"] = tf.math.count_nonzero(
        right_outliers) / sum(right_indexes)

    return metrics


@tf.autograph.experimental.do_not_convert
def create_gcnn(nodes_shape: int, edges_shape: int, channels: List[int],
                n_layers: int) -> Model:
    """Create a GCN model.

    Args:
        nodes_shape: dimension of node features
        edges_shape: dimension of edge features
        channels: number of channels per layer
        n_layers: number of layers

    Returns:
        model: GCN model

    """
    # Node features
    X = Input(shape=(None, nodes_shape))
    # Adjacency matrix
    A = Input(shape=(None, None))
    # Edge features
    E = Input(shape=(None, None, edges_shape))

    # Cast to float32
    x = Lambda(lambda x: tf.cast(x, tf.float32))(X)
    a = Lambda(lambda x: tf.cast(x, tf.float32))(A)
    e = Lambda(lambda x: tf.cast(x, tf.float32))(E)

    # Edge embedding layer / Apply linear transformation to edge features
    f = TimeDistributed(Dense(8, use_bias=False))(e)

    for i in range(n_layers):
        # ECCConv is a graph convolutional layer
        # reference: https://graphneural.network/layers/convolution/#eccconv
        x = ECCConv(channels[i], activation="tanh")([x, a, f])
        x = BatchNormalization()(x)

    # Global attention pooling layer
    x = GlobalAttnSumPool()(x)
    # Dense layer with leaky relu activation
    x = Dense(1024, LeakyReLU(alpha=0.01))(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization()(x)

    # Final layer with maximum likelihood estimation
    # return a pair (predicted affinity & error variance)
    output = MLEDense(1)(x)

    # Build model and return it
    return Model(inputs=[X, A, E], outputs=output)


class MLEDense(Layer):
    """Maximum likelihood estimation dense layer.

    Attributes:
        units: number of units

    """

    def __init__(self, units: int = 1):
        """ "
        Initialize the layer.

        Args:
            units: number of units

        """
        super(MLEDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        """Build the layer.

        Args:
            input_shape: shape of the input tensor

        """
        # pseudo-prior of variance
        init = tf.random_normal_initializer(mean=0.0, stddev=1.0)

        # weight
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            name="final_weight",
            trainable=True,
        )

        # bias
        self.b = self.add_weight(
            shape=(self.units, ),
            initializer="random_normal",
            name="final_bias",
            trainable=True,
        )

        # variance of error distribution
        self.sigma = tf.Variable(init(shape=(self.units, self.units)),
                                 name="sigma",
                                 trainable=True)

    def call(self, inputs):
        """Call the layer.

        Args:
            inputs: input tensor

        Returns:
            affinity, sigma: pair (predicted affinity & error variance)

        """
        affinity = tf.matmul(inputs, self.w) + self.b
        return tf.concat([affinity, self.sigma], axis=0)

    def get_config(self):
        """Get the layer configuration."""
        config = super(MLEDense, self).get_config()
        config.update({"units": self.units})
        return config
