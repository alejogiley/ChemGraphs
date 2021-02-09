import tensorflow as tf

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
from tensorflow.keras.models import Model
from tensorflow_probability.optimizers import LazyAdam
from spektral.data import BatchLoader
from spektral.layers import ECCConv, GlobalAttnSumPool

from gcnn.metrics import rsquared
from gcnn.utils import sigma


def train_model(
    dataset,
    tf_loss,
    n_layers=1,
    channels=[32],
    batch_size=32,
    number_epochs=40,
    learning_rate=0.001,
):
    """Comment

    Args:
        dataset: EstrogenDB instance
        tf_loss: tensorflow-type loss function
        n_layers: number of ECCConv layers in model
        channels: number of convolutional channels per layer
        batch_size: size of mini-batches
        number_epochs: number of epochs
        learning_rate: optimizer learning rate

    Returns:
        trained model and training history

    """

    # Parameters
    size_nodes = dataset.n_node_features  # Dimension of node features
    size_feats = dataset.n_edge_features  # Dimension of edge features

    # Create GCN model
    model = GCNN(
        nodes_shape=size_nodes,
        edges_shape=size_feats,
        channels=channels,
        n_layers=n_layers,
    )

    # Compile GCN
    model.compile(
        optimizer=LazyAdam(learning_rate),
        metrics=[rsquared, sigma],
        loss=tf_loss,
    )

    # Print network summary
    model.summary()

    # Loader returns batches of graphs
    # with zero-padding done batch-wise
    loader = BatchLoader(dataset, batch_size=batch_size, shuffle=True)

    # Trains the model
    history = model.fit(
        loader.load(),
        epochs=number_epochs,
        steps_per_epoch=loader.steps_per_epoch,
    )

    return model, history


def evaluate_model(model, test_set):

    # Loader returns batches of graphs
    # with zero-padding done batch-wise
    loader = BatchLoader(tests_set, batch_size=1, shuffle=False)

    # generate a pair (predicted affinity & sigma) per graph
    prediction = model.predict(loader.load(), steps=len(tests_set))

    # discarding sigma
    pred_values = prediction[::2]
    # experimental affinity values
    true_values = np.array(
        [tests_set[i]["y"][2] for i in range(tests_set.n_graphs)]
    )

    # censured data indexes
    lefts_indexes = np.array(
        [tests_set[i]["y"][0] for i in range(tests_set.n_graphs)]
    )
    right_indexes = np.array(
        [tests_set[i]["y"][1] for i in range(tests_set.n_graphs)]
    )
    # non-censored data indexes
    inner_indexes = (1 - right_indexes) * (1 - lefts_indexes)

    ##################################
    # Performance metrics
    ##################################

    metrics = {
        "mae": 0,
        "mse": 0,
        "pearson": 0,
        "rsquare": 0,
        "fraction_lefts_outliers": 0,
        "fraction_right_outliers": 0,
    }

    # estimate MEAN ABSOLUTE ERROR
    mae = tf.keras.losses.MeanAbsoluteError()
    metrics["mae"] = mae(
        true_values[(inner_indexes > 0)], pred_values[(inner_indexes > 0)]
    ).numpy()

    # estimate MEAN SQUARED ERROR
    mae = tf.keras.losses.MeanSquaredError()
    metrics["mse"] = mae(
        true_values[(inner_indexes > 0)], pred_values[(inner_indexes > 0)]
    ).numpy()

    # estimate PEARSON CORRELATION
    metrics["pearson"] = pearson(
        true_values[(inner_indexes > 0)], pred_values[(inner_indexes > 0)]
    )

    # estimate R^2 CORRELATION
    metrics["rsquared"] = rsquared(
        true_values[(inner_indexes > 0)], pred_values[(inner_indexes > 0)]
    )

    # estimate FRACTION of censored values
    # predicted higher than true boundaries
    lefts_outliers = tf.∏nn.relu(
        pred_values[(lefts_indexes > 0)] - true_values[(lefts_indexes > 0)]
    )
    right_outliers = tf.nn.relu(
        true_values[(right_indexes > 0)] - pred_values[(right_indexes > 0)]
    )

    metrics["fraction_lefts_outliers"] = tf.math.count_nonzero(
        lefts_outliers
    ) / sum(lefts_indexes)
    metrics["fraction_right_outliers"] = tf.math.count_nonzero(
        right_outliers
    ) / sum(right_indexes)

    return metrics


class MLEDense(Layer):
    def __init__(self, units=1):
        super(MLEDense, self).__init__()
        self.units = units

    def build(self, input_shape):

        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )

        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

        # pseudo-prior of variance
        init = tf.random_normal_initializer(mean=0.0, stddev=1.0)

        # variance of error distribution
        self.sigma = tf.Variable(
            init(shape=(self.units, self.units)), trainable=True
        )

    def call(self, inputs):
        y = tf.matmul(inputs, self.w) + self.b
        return tf.concat([y, self.sigma], axis=0)


class GCNN(Model):
    def __init__(self, nodes_shape, edges_shape, channels, n_layers, **kwargs):
        super(GCNN, self).__init__()

        # initialize operations
        self.dropout = Dropout(0.25)
        self.pooling = GlobalAttnSumPool()
        self.batchnm = BatchNormalization()

        self.convs = []
        self.batch = []
        # initialize convolutional
        # and batchnorm layers
        for i in range(n_layers):

            self.batch.append(BatchNormalization())

            self.convs.append(ECCConv(channels[i], activation="tanh"))

        # embedding for (ohc) edges features
        self.embedding = TimeDistributed(Dense(8, use_bias=False))

        # initialize dense layers
        self.dense = Dense(1024, LeakyReLU(alpha=0.01))
        # last layer linear model: y = ax + b
        self.linear = MLEDense(1)

        # format layers
        self.format = Lambda(lambda x: tf.cast(x, tf.float32))

        # Parameters of the model
        self.X = Input(shape=(None, nodes_shape))
        self.A = Input(shape=(None, None))
        self.E = Input(shape=(None, None, edges_shape))

        self.inp = [self.X, self.A, self.E]
        self.out = self.call(self.inp)
        super(GCNN, self).__init__(inputs=self.inp, outputs=self.out, **kwargs)

    def build(self):

        self._is_graph_network = True
        self._init_graph_network(inputs=self.inp, outputs=self.out)

    def call(self, input, **kwargs):

        x, a, e = input
        x = self.format(x)
        a = self.format(a)
        e = self.format(e)
        e = self.embedding(e)

        for conv, batch in zip(self.convs, self.batch):
            x = batch(conv([x, a, e]))

        # polling nodes
        x = self.pooling(x)

        # MLP block
        x = self.dense(x)
        x = self.dropout(x)
        x = self.batchnm(x)

        # prediction + error
        return self.linear(x)