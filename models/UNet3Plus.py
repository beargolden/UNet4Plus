# coding=utf-8
from keras.layers import *
from keras.models import *


def conv_block(input_tensor, num_filters):
    out = Conv2D(num_filters, kernel_size=3, padding="same", activation="relu",
                 kernel_initializer="he_normal")(input_tensor)
    out = Conv2D(num_filters, kernel_size=3, padding="same", activation="relu",
                 kernel_initializer="he_normal")(out)
    return out


def conv(input_tensor, num_filters):
    out = Conv2D(num_filters, kernel_size=3, padding="same", activation="relu",
                 kernel_initializer="he_normal")(input_tensor)
    return out


def up_conv(input_tensor, num_filters, up_size):
    out = UpSampling2D(up_size, interpolation="bilinear")(input_tensor)
    out = Conv2D(num_filters, kernel_size=3, padding="same", activation="relu",
                 kernel_initializer="he_normal")(out)
    return out


def down_conv(input_tensor, num_filters, down_size):
    out = MaxPooling2D(down_size)(input_tensor)
    out = Conv2D(num_filters, kernel_size=3, padding="same", activation="relu",
                 kernel_initializer="he_normal")(out)
    return out


def UNet3Plus_w_DeepSupv(num_classes, input_height, input_width, num_filters):
    """
    U-Net3+ with deep supervision
    Paper : https://arxiv.org/abs/2004.08790
    """
    inputs = Input(shape=(input_height, input_width, 1))

    filters = [num_filters, num_filters * 2, num_filters * 4, num_filters * 8, num_filters * 16]

    e1 = conv_block(inputs, filters[0])

    e2 = MaxPooling2D()(e1)
    e2 = conv_block(e2, filters[1])

    e3 = MaxPooling2D()(e2)
    e3 = conv_block(e3, filters[2])

    e4 = MaxPooling2D()(e3)
    e4 = conv_block(e4, filters[3])

    e5 = MaxPooling2D()(e4)
    e5 = conv_block(e5, filters[4])

    e1_d_d4 = down_conv(e1, filters[0], 8)
    e2_d_d4 = down_conv(e2, filters[0], 4)
    e3_d_d4 = down_conv(e3, filters[0], 2)
    e4_d4 = conv(e4, filters[0])
    e5_u_d4 = up_conv(e5, filters[0], 2)
    d4 = Concatenate()([e1_d_d4, e2_d_d4, e3_d_d4, e4_d4, e5_u_d4])
    d4 = conv_block(d4, filters[0] * 5)

    e1_d_d3 = down_conv(e1, filters[0], 4)
    e2_d_d3 = down_conv(e2, filters[0], 2)
    e3_d3 = conv(e3, filters[0])
    e5_u_d3 = up_conv(e5, filters[0], 4)
    d4_u_d3 = up_conv(d4, filters[0], 2)
    d3 = Concatenate()([e1_d_d3, e2_d_d3, e3_d3, e5_u_d3, d4_u_d3])
    d3 = conv_block(d3, filters[0] * 5)

    e1_d_d2 = down_conv(e1, filters[0], 2)
    e2_d2 = conv(e2, filters[0])
    e5_u_d2 = up_conv(e5, filters[0], 8)
    d4_u_d2 = up_conv(d4, filters[0], 4)
    d3_u_d2 = up_conv(d3, filters[0], 2)
    d2 = Concatenate()([e1_d_d2, e2_d2, e5_u_d2, d4_u_d2, d3_u_d2])
    d2 = conv_block(d2, filters[0] * 5)

    e1_d1 = conv(e1, filters[0])
    e5_u_d1 = up_conv(e5, filters[0], 16)
    d4_u_d1 = up_conv(d4, filters[0], 8)
    d3_u_d1 = up_conv(d3, filters[0], 4)
    d2_u_d1 = up_conv(d2, filters[0], 2)
    d1 = Concatenate()([e1_d1, e5_u_d1, d4_u_d1, d3_u_d1, d2_u_d1])
    d1 = conv_block(d1, filters[0] * 5)

    outputs = Average()([d1,
                         up_conv(d2, filters[0] * 5, 2),
                         up_conv(d3, filters[0] * 5, 4),
                         up_conv(d4, filters[0] * 5, 8)])
    outputs = Conv2D(num_classes, kernel_size=3, padding="same", activation="sigmoid",
                     kernel_initializer="he_normal")(outputs)

    model = Model(inputs=inputs, outputs=outputs, name="UNet3Plus_w_DeepSupv")

    return model


def UNet3Plus_wo_DeepSupv(num_classes, input_height, input_width, num_filters):
    """
    U-Net3+ without deep supervision
    Paper : https://arxiv.org/abs/2004.08790
    """
    inputs = Input(shape=(input_height, input_width, 1))

    filters = [num_filters, num_filters * 2, num_filters * 4, num_filters * 8, num_filters * 16]

    e1 = conv_block(inputs, filters[0])

    e2 = MaxPooling2D()(e1)
    e2 = conv_block(e2, filters[1])

    e3 = MaxPooling2D()(e2)
    e3 = conv_block(e3, filters[2])

    e4 = MaxPooling2D()(e3)
    e4 = conv_block(e4, filters[3])

    e5 = MaxPooling2D()(e4)
    e5 = conv_block(e5, filters[4])

    e1_d_d4 = down_conv(e1, filters[0], 8)
    e2_d_d4 = down_conv(e2, filters[0], 4)
    e3_d_d4 = down_conv(e3, filters[0], 2)
    e4_d4 = conv(e4, filters[0])
    e5_u_d4 = up_conv(e5, filters[0], 2)
    d4 = Concatenate()([e1_d_d4, e2_d_d4, e3_d_d4, e4_d4, e5_u_d4])
    d4 = conv_block(d4, filters[0] * 5)

    e1_d_d3 = down_conv(e1, filters[0], 4)
    e2_d_d3 = down_conv(e2, filters[0], 2)
    e3_d3 = conv(e3, filters[0])
    e5_u_d3 = up_conv(e5, filters[0], 4)
    d4_u_d3 = up_conv(d4, filters[0], 2)
    d3 = Concatenate()([e1_d_d3, e2_d_d3, e3_d3, e5_u_d3, d4_u_d3])
    d3 = conv_block(d3, filters[0] * 5)

    e1_d_d2 = down_conv(e1, filters[0], 2)
    e2_d2 = conv(e2, filters[0])
    e5_u_d2 = up_conv(e5, filters[0], 8)
    d4_u_d2 = up_conv(d4, filters[0], 4)
    d3_u_d2 = up_conv(d3, filters[0], 2)
    d2 = Concatenate()([e1_d_d2, e2_d2, e5_u_d2, d4_u_d2, d3_u_d2])
    d2 = conv_block(d2, filters[0] * 5)

    e1_d1 = conv(e1, filters[0])
    e5_u_d1 = up_conv(e5, filters[0], 16)
    d4_u_d1 = up_conv(d4, filters[0], 8)
    d3_u_d1 = up_conv(d3, filters[0], 4)
    d2_u_d1 = up_conv(d2, filters[0], 2)
    d1 = Concatenate()([e1_d1, e5_u_d1, d4_u_d1, d3_u_d1, d2_u_d1])
    d1 = conv_block(d1, filters[0] * 5)

    outputs = Conv2D(num_classes, kernel_size=3, padding="same", activation="sigmoid",
                     kernel_initializer="he_normal")(d1)

    model = Model(inputs=inputs, outputs=outputs, name="UNet3Plus_wo_DeepSupv")

    return model
