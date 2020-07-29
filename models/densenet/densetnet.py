from tensorflow.keras.layers import Input, Activation, Conv2D, Dropout, BatchNormalization, AveragePooling2D, \
    Concatenate, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def DenseNet():
    input_size = (256, 256, 3)
    growth_rate = 16
    weight_decay = 1e-4
    dense_block_layers = [4, 5, 7, 10, 12, 15]
    dropout = 0.2
    compression = 1.0
    bottleneck = True
    nb_filters = 48
    skip_connections = []

    input_layer = Input(input_size)
    x = Conv2D(nb_filters, (3, 3), padding="same", use_bias=False, kernel_regularizer=l2(weight_decay))(input_layer)

    for i, block_size in enumerate(dense_block_layers):
        x, nb_filters = _dense_block(x, block_size, nb_filters, growth_rate, dropout, bottleneck, weight_decay)

        skip_connections.append(x)
        if i < len(dense_block_layers) - 1:
            x, nb_filters = _transition_down_layer(x, nb_filters, dropout, compression, weight_decay)

    skip_connections = skip_connections[::-1][1:]

    for i, block_size in enumerate(dense_block_layers[::-1][1:]):
        x, nb_filters = _transition_up_layer(skip_connections[i], x, nb_filters)

        x, nb_filters = _dense_block(x, block_size, nb_filters, growth_rate, dropout, bottleneck, weight_decay)

    x = Conv2D(6, (1, 1), activation="softmax", kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(
        x)

    return Model(inputs=input_layer, outputs=x)


def _convolution_block(x, nb_filters, dropout_rate=None, bottleneck=False, weight_decay=1e-4):
    if bottleneck:
        x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
        x = Activation("relu")(x)
        x = Conv2D(nb_filters * 4, (1, 1), padding="same", use_bias=False, kernel_regularizer=l2(weight_decay))(x)

        if dropout_rate:
            x = Dropout(dropout_rate)(x)

    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation("relu")(x)
    x = Conv2D(nb_filters, (3, 3), padding="same", use_bias=False, kernel_regularizer=l2(weight_decay))(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def _dense_block(x, nb_layers, nb_filters, growth_rate, dropout_rate=None, bottleneck=False, weight_decay=1e-4):
    for _ in range(nb_layers):
        block = _convolution_block(x, growth_rate, dropout_rate, bottleneck, weight_decay)
        x = Concatenate(axis=3)([x, block])
        nb_filters += growth_rate

    return x, nb_filters


def _transition_down_layer(x, nb_filters, dropout_rate=None, compression=1.0, weight_decay=1e-4):
    nb_filters = int(nb_filters * compression)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation("relu")(x)
    x = Conv2D(nb_filters, (1, 1), padding="same", use_bias=False, kernel_regularizer=l2(weight_decay))(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x, nb_filters


def _transition_up_layer(skip_connection, x, nb_filters):
    x = Conv2DTranspose(nb_filters, (2, 2), strides=(2, 2))(x)
    x = Concatenate(axis=3)([x, skip_connection])
    return x, nb_filters
