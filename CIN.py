import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten
from tensorflow.keras.models import Model


def CIN_model_creator(input_dim, hidden_layers):
    # create the input layer
    input_layer = Input(shape=(input_dim,))
    embedding_layer = Dense(3, activation='relu')(input_layer)
    hidden_layers_list = [embedding_layer]

    for i in range(hidden_layers):
        # iteratively building multiple hidden layers
        feature_maps = []
        for j in range(input_dim):
            feature_map = Dense(3, activation='relu')(hidden_layers_list[-1])
            feature_maps.append(feature_map)
        interactions = []
        for j in range(input_dim):
            for k in range(j + 1, input_dim):
                interaction = tf.multiply(feature_maps[j], feature_maps[k])
                interactions.append(interaction)
        hidden_layer = Concatenate()(interactions)
        hidden_layers_list.append(hidden_layer)

    # flatten the output layer
    output_layer = Flatten()(hidden_layers_list[-1])
    output_layer = Dense(1, activation='sigmoid')(output_layer)
    model = Model(inputs=input_layer, outputs=output_layer)

    return model
