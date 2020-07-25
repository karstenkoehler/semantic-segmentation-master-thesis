from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Cropping2D, concatenate
from tensorflow.keras.models import Model


def UNet(input_size=(572, 572, 3)):
    x = Input(input_size)

    x, skip1 = _downsampling_block(x, 64)
    x, skip2 = _downsampling_block(x, 128)
    x, skip3 = _downsampling_block(x, 256)
    x, skip4 = _downsampling_block(x, 512)

    x = Conv2D(1024, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(x)
    x = Conv2D(1024, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(x)
    x = Dropout(0.5)(x)

    x = _upsampling_block(x, skip4, 512, 4)
    x = _upsampling_block(x, skip3, 256, 16)
    x = _upsampling_block(x, skip2, 128, 40)
    x = _upsampling_block(x, skip1, 64, 88)

    out = Conv2D(6, 1, activation='sigmoid')(x)
    return Model(inputs=x, outputs=out)


def _downsampling_block(x, nb_filters):
    x = Conv2D(nb_filters, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(x)
    skip = Conv2D(nb_filters, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(x)
    x = MaxPooling2D(pool_size=(2, 2))(skip)
    return x, skip


def _upsampling_block(x, skip, nb_filters, crop_px):
    skip = Cropping2D(crop_px)(skip)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(nb_filters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = concatenate([skip, x], axis=3)
    x = Conv2D(nb_filters, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(x)
    x = Conv2D(nb_filters, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(x)
    return x


if __name__ == '__main__':
    model = UNet()
    model.summary()
