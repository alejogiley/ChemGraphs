from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Dense, Input,
    Activation, Dropout,
    BatchNormalization
)

from spektral.data import BatchLoader
from spektral.transforms import LayerPreprocess
from spektral.layers import (
    ECCConv, GCSConv,
    MinCutPool, GlobalSumPool
)

def gcn_model(nodes_shape, edges_shape, n_channels, n_layers, n_neurons):
    
    X = Input(shape=(None, nodes_shape))
    A = Input(shape=(None, None))
    E = Input(shape=(None, None, edges_shape))

    y = ECCConv(n_channels[0])([X, A, E])
    y = Activation('relu')(y)
    
    for i in range(n_layers - 1):
        y = ECCConv(n_channels[i + 1])([y, A, E])
        y = BatchNormalization(renorm=True)(y)
        y = Activation('relu')(y)
        y = Dropout(0.05)(y)
    
    # prediction
    y = GlobalSumPool()(y)
    y = Dense(n_neurons)(y)
    y = Activation('relu')(y)
    y = Dropout(0.25)(y)
    O = Dense(2)(y)
    
    return Model(inputs=[X, A, E], outputs=O)


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