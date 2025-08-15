# coding=utf-8
from keras.layers import *
from keras.models import *

USE_MAXPOOLING = True
USE_UPSAMPLING = True


def conv_block(input_tensor, num_filters):
    out = Conv2D(num_filters, kernel_size=3, padding="same", activation="relu",
                 kernel_initializer="he_normal")(input_tensor)
    out = Conv2D(num_filters, kernel_size=3, padding="same", activation="relu",
                 kernel_initializer="he_normal")(out)
    return out


def down_maxpool(input_tensor):
    out = MaxPooling2D(pool_size=2)(input_tensor)
    return out


def down_stridedconv(input_tensor, num_filters):
    out = Conv2D(num_filters, kernel_size=3, strides=2, padding="same", activation="relu",
                 kernel_initializer="he_normal")(input_tensor)
    return out


def up_conv(input_tensor, num_filters):
    out = UpSampling2D(size=2, interpolation="bilinear")(input_tensor)
    out = Conv2D(num_filters, kernel_size=3, padding="same", activation="relu",
                 kernel_initializer="he_normal")(out)
    return out


def up_transposedconv(input_tensor, num_filters):
    out = Conv2DTranspose(num_filters, kernel_size=3, strides=2, padding="same", activation="relu",
                          kernel_initializer="he_normal")(input_tensor)
    return out


def UNet(num_classes, input_height, input_width, num_filters):
    """
    U-Net
    Paper : https://arxiv.org/abs/1505.04597
    """
    inputs = Input(shape=(input_height, input_width, 1))

    filters = [num_filters, num_filters * 2, num_filters * 4, num_filters * 8, num_filters * 16]

    e1 = conv_block(inputs, filters[0])

    e2 = down_maxpool(e1) if USE_MAXPOOLING else down_stridedconv(e1, filters[0])
    e2 = conv_block(e2, filters[1])

    e3 = down_maxpool(e2) if USE_MAXPOOLING else down_stridedconv(e2, filters[1])
    e3 = conv_block(e3, filters[2])

    e4 = down_maxpool(e3) if USE_MAXPOOLING else down_stridedconv(e3, filters[2])
    e4 = conv_block(e4, filters[3])

    e5 = down_maxpool(e4) if USE_MAXPOOLING else down_stridedconv(e4, filters[3])
    e5 = conv_block(e5, filters[4])

    d4 = up_conv(e5, filters[3]) if USE_UPSAMPLING else up_transposedconv(e5, filters[3])
    d4 = Concatenate()([e4, d4])
    d4 = conv_block(d4, filters[3])

    d3 = up_conv(d4, filters[2]) if USE_UPSAMPLING else up_transposedconv(d4, filters[2])
    d3 = Concatenate()([e3, d3])
    d3 = conv_block(d3, filters[2])

    d2 = up_conv(d3, filters[1]) if USE_UPSAMPLING else up_transposedconv(d3, filters[1])
    d2 = Concatenate()([e2, d2])
    d2 = conv_block(d2, filters[1])

    d1 = up_conv(d2, filters[0]) if USE_UPSAMPLING else up_transposedconv(d2, filters[0])
    d1 = Concatenate()([e1, d1])
    d1 = conv_block(d1, filters[0])

    outputs = Conv2D(num_classes, kernel_size=3, padding="same", activation="sigmoid",
                     kernel_initializer="he_normal")(d1)

    model = Model(inputs=inputs, outputs=outputs, name="UNet")

    return model
