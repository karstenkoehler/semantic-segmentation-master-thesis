from tensorflow.keras.models import Model

from models.unet.unet import UNet


def WNet(feature_maps=None, nb_classes=6, dropout=0.65, model_name_suffix=""):
    if feature_maps is None:
        feature_maps = [64, 128, 256, 512, 1024]

    encoder_input, encoder_output = UNet(input_size=(256, 256, 3), feature_maps=feature_maps, nb_classes=nb_classes,
                                         dropout=dropout, conv_padding="same", build_model=False)
    _, decoder_output = UNet(input_layer=encoder_output, nb_classes=3, output_layer_activation="sigmoid",
                             dropout=dropout, feature_maps=feature_maps, conv_padding="same", build_model=False)

    nb_conv_layers = 6 + 10 * (len(feature_maps) - 1)
    dropout_suffix = "D" if dropout > 0.0 else ""
    model_name = f"WNet-{nb_conv_layers}{dropout_suffix}-{nb_classes}{model_name_suffix}"
    full_model = Model(name=model_name, inputs=encoder_input, outputs=decoder_output)
    encoder_model = Model(name=f"{model_name}-Encoder", inputs=encoder_input, outputs=encoder_output)
    return full_model, encoder_model


if __name__ == '__main__':
    model, encoder = WNet()
    model.summary()
