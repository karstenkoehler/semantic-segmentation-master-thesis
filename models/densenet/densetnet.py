from tensorflow.keras.layers import Input, Activation, Conv2D, Dropout, BatchNormalization, AveragePooling2D, \
    Concatenate, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def DenseNet(dense_block_layers=None, growth_rate=16, initial_nb_filters=48, compression=1.0, dropout=0.2,
             model_name_suffix="", input_size=(256, 256, 3)):
    nb_filters = initial_nb_filters
    nb_conv_layers = 0

    if dense_block_layers is None:
        dense_block_layers = [4, 5, 7, 10, 12, 15]
    skip_connections = []

    input_layer = Input(input_size)
    x = Conv2D(nb_filters, (3, 3), padding="same", use_bias=False, kernel_regularizer=_l2reg())(input_layer)
    nb_conv_layers += 1

    for block_size in dense_block_layers:
        dense_block_input = x
        x, nb_filters = _dense_block(x, block_size, nb_filters, growth_rate, dropout)
        x = Concatenate(axis=3)([dense_block_input, x])
        skip_connections.append(x)
        x, nb_filters = _transition_down_layer(x, nb_filters, dropout, compression)

    skip_connections = skip_connections[::-1]

    for i, block_size in enumerate(dense_block_layers[::-1][1:]):
        x, nb_filters = _transition_up_layer(skip_connections[i], x, nb_filters)
        x, nb_filters = _dense_block(x, block_size, nb_filters, growth_rate, dropout)

    x = Conv2D(6, (1, 1), activation="softmax", kernel_regularizer=_l2reg(), bias_regularizer=_l2reg())(x)
    nb_conv_layers += 1

    return Model(inputs=input_layer, outputs=x)


def _convolution_block(x, nb_filters, dropout):
    x = BatchNormalization(gamma_regularizer=_l2reg(), beta_regularizer=_l2reg())(x)
    x = Activation("relu")(x)
    x = Conv2D(nb_filters, (3, 3), padding="same", use_bias=False, kernel_regularizer=_l2reg())(x)

    if dropout > 0:
        x = Dropout(dropout)(x)

    return x


def _dense_block(x, nb_layers, nb_filters, growth_rate, dropout):
    block_layers = []
    for _ in range(nb_layers):
        layer = _convolution_block(x, growth_rate, dropout)
        block_layers.append(layer)
        x = Concatenate(axis=3)([x, layer])
        nb_filters += growth_rate


    return Concatenate(axis=3)(block_layers), nb_filters


def _transition_down_layer(x, nb_filters, dropout_rate=None, compression=1.0):
    nb_filters = int(nb_filters * compression)
    x = BatchNormalization(gamma_regularizer=_l2reg(), beta_regularizer=_l2reg())(x)
    x = Activation("relu")(x)
    x = Conv2D(nb_filters, (1, 1), padding="same", use_bias=False, kernel_regularizer=_l2reg())(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x, nb_filters


def _transition_up_layer(skip_connection, x, nb_filters):
    x = Conv2DTranspose(nb_filters, (2, 2), strides=(2, 2))(x)
    x = Concatenate(axis=3)([x, skip_connection])
    return x, nb_filters


def _l2reg(weight_decay=1e-4):
    return l2(weight_decay)


if __name__ == '__main__':
    model = DenseNet()
    model.summary()
