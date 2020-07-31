from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Cropping2D, concatenate
from tensorflow.keras.models import Model


def UNet(feature_maps=None, nb_classes=6, dropout=0.5, model_name_suffix="", output_layer_activation="softmax",
         input_layer=None, input_size=None, conv_padding="valid"):
    nb_conv_layers = 0

    if feature_maps is None:
        feature_maps = [64, 128, 256, 512, 1024]
    skip_connections = []

    if input_layer is None and input_size is None:
        raise ValueError("UNet: either input_layer or input_size has to be specified")

    if input_layer is None:
        input_layer = Input(input_size)

    x = input_layer
    for nb_filters in feature_maps[:-1]:
        x, skip = _downsampling_block(x, nb_filters, conv_padding)
        skip_connections += [skip]
        nb_conv_layers += 2

    skip_connections = skip_connections[::-1]
    x = Conv2D(feature_maps[-1], 3, activation='relu', padding=conv_padding, kernel_initializer='he_normal')(x)
    x = Conv2D(feature_maps[-1], 3, activation='relu', padding=conv_padding, kernel_initializer='he_normal')(x)
    nb_conv_layers += 2

    if dropout > 0.0:
        x = Dropout(dropout)(x)

    for i, nb_filters in enumerate(feature_maps[:-1][::-1]):
        crop_px = (skip_connections[i].shape[1] - (x.shape[1] * 2)) // 2
        x = _upsampling_block(x, skip_connections[i], nb_filters, crop_px, conv_padding)
        nb_conv_layers += 3

    output_layer = Conv2D(nb_classes, 1, activation=output_layer_activation, padding=conv_padding)(x)
    nb_conv_layers += 1

    model_name = f"unet-{nb_conv_layers}{'D' if dropout > 0.0 else ''}{model_name_suffix}"
    return Model(inputs=input_layer, outputs=output_layer, name=model_name), (input_layer, output_layer)


def _downsampling_block(x, nb_filters, conv_padding):
    x = Conv2D(nb_filters, 3, activation='relu', padding=conv_padding, kernel_initializer='he_normal')(x)
    skip = Conv2D(nb_filters, 3, activation='relu', padding=conv_padding, kernel_initializer='he_normal')(x)
    x = MaxPooling2D(pool_size=(2, 2))(skip)
    return x, skip


def _upsampling_block(x, skip, nb_filters, crop_px, conv_padding):
    skip = Cropping2D(crop_px)(skip)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(nb_filters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = concatenate([skip, x], axis=3)
    x = Conv2D(nb_filters, 3, activation='relu', padding=conv_padding, kernel_initializer='he_normal')(x)
    x = Conv2D(nb_filters, 3, activation='relu', padding=conv_padding, kernel_initializer='he_normal')(x)
    return x


if __name__ == '__main__':
    model, _ = UNet(input_size=(572, 572, 3))
    model.summary()
