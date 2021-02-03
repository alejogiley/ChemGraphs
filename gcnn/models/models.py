from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Dense, Input, Lambda,
    Activation, Dropout,
    BatchNormalization
)

from spektral.data import BatchLoader
from spektral.transforms import LayerPreprocess
from spektral.layers import (
    ECCConv, GCSConv,
    MinCutPool, GlobalSumPool
)


def train_model(dataset, epochs, learning_rate, n_channels, n_layers, n_neurons): 
    
    # Parameters
    F = dataset.n_node_features  # Dimension of node features
    S = dataset.n_edge_features  # Dimension of edge features

    # Create GCN model
    model = gcn_model(
        nodes_shape=F, 
        edges_shape=S, 
        n_layers=n_layers, 
        n_neurons=n_neurons,
        n_channels=n_channels, 
    )
    
    # Compile GCN
    model.compile(
        optimizer=Adam(lr=learning_rate), 
        metrics=["mae"],
        loss="mse")
    
    # Print network summary
    model.summary()
    
    # Loader returns batches of graphs
    # with zero-padding done batch-wise
    loader = BatchLoader(
        dataset, 
        batch_size=batch_size)
    
    # Trains the model
    history = model.fit(
        loader.load(),
        epochs=epochs,
        steps_per_epoch=loader.steps_per_epoch)
    
    return model, history


def tobit_loss(y_true, y_pred, sigma, eps=1e-7):

    # indicators of left-, right-censoring
    y_lefts = y_true[:, 0]
    y_right = y_true[:, 1]
    y_value = y_true[:, 2]

    # normal distribution
    normal = tfp.distributions.Normal(loc=0., scale=1.)

    # probability function of normal distribution at point y_value
    prob = normal.prob((y_value - y_pred) / sigma) / sigma
    # probability of point random variable being > than y_value
    right_prob = 1 - normal.cdf((y_value - y_pred) / sigma)
    # probability of random variable being < than y_value
    lefts_prob = normal.cdf((y_value - y_pred) / sigma)

    # clip tensor values
    prob = tf.clip_by_value(
        prob,
        clip_value_min=eps,
        clip_value_max=1/eps)

    right_prob = tf.clip_by_value(
        right_prob,
        clip_value_min=eps,
        clip_value_max=1/eps)

    left_prob = tf.clip_by_value(
        lefts_prob,
        clip_value_min=eps,
        clip_value_max=1/eps)

    # logarithm of likelihood
    logp = tf.math.log(prob) * (1 - y_right) * (1 - y_lefts) \
           + tf.math.log(right_prob) * y_right * (1 - y_lefts) \
           + tf.math.log(lefts_prob) * y_lefts * (1 - y_right)

    return - tf.reduce_sum(logp)


def mask_loss(y_true, y_pred):

    # y_pred = y_pred + sigma
    # add sigma to product
    y_pred = y_pred[:-1] + y_pred[-1]

    # indicators of left-, right-censoring
    y_lefts = tf.cast(y_true[:, 0], dtype=tf.int32)
    y_right = tf.cast(y_true[:, 1], dtype=tf.int32)

    mask = (1 - y_right) * (1 - y_lefts)
    mask = tf.logical_not(tf.equal(mask, 0))

    true = tf.boolean_mask(y_true[:, 2], mask)
    pred = tf.boolean_mask(y_pred[:, 0], mask)
    return tf.reduce_mean(tf.square(true - pred))


def mask_pearson(y_true, y_pred):
    """

    r^2 = Cov(X, Y)^2 / VAR(X) * VAR(Y)
        = (E[XY] - E[X]E[Y])^2 / (E[X^2] - E[X]^2) * (E[Y^2] - E[Y]^2)

    """
    # y_pred = y_pred + sigma
    # add sigma to product
    y_pred = y_pred[:-1] + y_pred[-1]

    # indicators of left-, right-censoring
    y_lefts = tf.cast(y_true[:, 0], dtype=tf.int32)
    y_right = tf.cast(y_true[:, 1], dtype=tf.int32)

    mask = (1 - y_right) * (1 - y_lefts)
    mask = tf.logical_not(tf.equal(mask, 0))

    true = tf.boolean_mask(y_true[:, 2], mask)
    pred = tf.boolean_mask(y_pred[:, 0], mask)

    # r2
    mean_x = tf.reduce_mean(true)
    mean_y = tf.reduce_mean(pred)

    mean_x2 = tf.reduce_mean(true * true)
    mean_y2 = tf.reduce_mean(pred * pred)
    mean_xy = tf.reduce_mean(true * pred)

    numerator = mean_xy - mean_x * mean_y
    numerator *= numerator

    denominator = mean_x2 - mean_x * mean_x
    denominator *= mean_y2 - mean_y * mean_y

    r2 = 0.0
    if denominator > 0.0:
        r2 = numerator / denominator

    return r2


class SimpleDense(tf.keras.layers.Layer):

    def __init__(self, units=1):
        
        super(SimpleDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True)
        
        self.b = self.add_weight(
            shape=(self.units, self.units),
            initializer='random_normal',
            trainable=True)

    def call(self, inputs):
        y = tf.matmul(inputs, self.w)
        return tf.concat([y, self.b], 0)


class GCNN(tf.keras.Model):

    def __init__(self, nodes_shape, edges_shape, channels, n_layers, n_neurons, **kwargs):
        
        super(GCNN, self).__init__()

        # initialize dense layers
        self.dense1 = Dense(n_neurons, "relu")
        self.dense2 = Dense(8, "relu")

        # initialize operations
        self.dropout = Dropout(0.15)
        self.pooling = GlobalSumPool()
        self.batchnm = BatchNormalization()

        # first block 
        self.conv1 = ECCConv(
            channels[0],
            kernel_network=[4, 8],
            activation='relu',))
        self.batch1 = BatchNormalization()

        # initialize edge-conditioned
        # convolutional layers
        self.convs = []
        self.batch = []
        for i in range(1, n_layers):
            self.batch.append(
                BatchNormalization())
            self.convs.append(
                ECCConv(channels[i],
                        kernel_network=[4, 8],
                        activation='relu',))

        # last layer linear model: y = ax + b
        self.linear = SimpleDense(1)

        # format layers
        self.type = Lambda(lambda x: tf.cast(x, tf.float32))

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
        a = self.type(a)

        x = self.batch1(self.conv1([x, a, e]))
        for conv, batch in zip(self.convs, self.batch):
            x = batch(conv([x, a, e]))

        # Max node features
        x = self.pooling(x)

        # MLP block
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.batchnm(x)
        x = self.dense2(x)

        return self.linear(x)
