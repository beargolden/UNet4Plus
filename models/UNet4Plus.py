# coding=utf-8
from keras.layers import *
from keras.models import *


def conv_block(input_tensor, num_filters):
    out = Conv2D(num_filters, kernel_size=3, padding="same", activation="relu", kernel_initializer="he_normal")(input_tensor)
    out = Conv2D(num_filters, kernel_size=3, padding="same", activation="relu", kernel_initializer="he_normal")(out)
    return out


def conv(input_tensor, num_filters):
    out = Conv2D(num_filters, kernel_size=3, padding="same", activation="relu", kernel_initializer="he_normal")(input_tensor)
    return out


def up_conv(input_tensor, num_filters, up_size):
    out = UpSampling2D(up_size, interpolation="bilinear")(input_tensor)
    out = Conv2D(num_filters, kernel_size=3, padding="same", activation="relu", kernel_initializer="he_normal")(out)
    return out


def down_conv(input_tensor, num_filters, down_size):
    out = MaxPooling2D(down_size)(input_tensor)
    out = Conv2D(num_filters, kernel_size=3, padding="same", activation="relu", kernel_initializer="he_normal")(out)
    return out


def UNet4Plus_w_DeepSupv(num_classes, input_height, input_width, num_filters):
    """
    U-Net4+ with deep supervision
    """
    inputs = Input(shape=(input_height, input_width, 1))

    e1 = conv_block(inputs, num_filters)

    e2 = down_conv(e1, num_filters, 2)
    e2 = conv_block(e2, num_filters)

    e1_d_e3 = down_conv(e1, num_filters, 4)
    e2_d_e3 = down_conv(e2, num_filters, 2)
    e3 = Concatenate()([e1_d_e3, e2_d_e3])
    e3 = conv_block(e3, num_filters * 2)

    e1_d_e4 = down_conv(e1, num_filters, 8)
    e2_d_e4 = down_conv(e2, num_filters, 4)
    e3_d_e4 = down_conv(e3, num_filters, 2)
    e4 = Concatenate()([e1_d_e4, e2_d_e4, e3_d_e4])
    e4 = conv_block(e4, num_filters * 3)

    e1_d_e5 = down_conv(e1, num_filters, 16)
    e2_d_e5 = down_conv(e2, num_filters, 8)
    e3_d_e5 = down_conv(e3, num_filters, 4)
    e4_d_e5 = down_conv(e4, num_filters, 2)
    e5 = Concatenate()([e1_d_e5, e2_d_e5, e3_d_e5, e4_d_e5])
    e5 = conv_block(e5, num_filters * 4)

    e1_d_d4 = down_conv(e1, num_filters, 8)
    e2_d_d4 = down_conv(e2, num_filters, 4)
    e3_d_d4 = down_conv(e3, num_filters, 2)
    e4_d4 = conv(e4, num_filters)
    e5_u_d4 = up_conv(e5, num_filters, 2)
    d4 = Concatenate()([e1_d_d4, e2_d_d4, e3_d_d4, e4_d4, e5_u_d4])
    d4 = conv_block(d4, num_filters * 5)

    e1_d_d3 = down_conv(e1, num_filters, 4)
    e2_d_d3 = down_conv(e2, num_filters, 2)
    e3_d3 = conv(e3, num_filters)
    e4_u_d3 = up_conv(e4, num_filters, 2)
    e5_u_d3 = up_conv(e5, num_filters, 4)
    d4_u_d3 = up_conv(d4, num_filters, 2)
    d3 = Concatenate()([e1_d_d3, e2_d_d3, e3_d3, e4_u_d3, e5_u_d3, d4_u_d3])
    d3 = conv_block(d3, num_filters * 6)

    e1_d_d2 = down_conv(e1, num_filters, 2)
    e2_d2 = conv(e2, num_filters)
    e3_u_d2 = up_conv(e3, num_filters, 2)
    e4_u_d2 = up_conv(e4, num_filters, 4)
    e5_u_d2 = up_conv(e5, num_filters, 8)
    d4_u_d2 = up_conv(d4, num_filters, 4)
    d3_u_d2 = up_conv(d3, num_filters, 2)
    d2 = Concatenate()([e1_d_d2, e2_d2, e3_u_d2, e4_u_d2, e5_u_d2, d4_u_d2, d3_u_d2])
    d2 = conv_block(d2, num_filters * 7)

    e1_d1 = conv(e1, num_filters)
    e2_u_d1 = up_conv(e2, num_filters, 2)
    e3_u_d1 = up_conv(e3, num_filters, 4)
    e4_u_d1 = up_conv(e4, num_filters, 8)
    e5_u_d1 = up_conv(e5, num_filters, 16)
    d4_u_d1 = up_conv(d4, num_filters, 8)
    d3_u_d1 = up_conv(d3, num_filters, 4)
    d2_u_d1 = up_conv(d2, num_filters, 2)
    d1 = Concatenate()([e1_d1, e2_u_d1, e3_u_d1, e4_u_d1, e5_u_d1, d4_u_d1, d3_u_d1, d2_u_d1])
    d1 = conv_block(d1, num_filters * 8)

    outputs = Concatenate()([d1,
                             UpSampling2D(size=2, interpolation="bilinear")(d2),
                             UpSampling2D(size=4, interpolation="bilinear")(d3),
                             UpSampling2D(size=8, interpolation="bilinear")(d4)])
    outputs = Conv2D(num_classes, kernel_size=3, padding="same", activation="sigmoid",
                     kernel_initializer="he_normal")(outputs)

    model = Model(inputs=inputs, outputs=outputs, name="U-Net4+ with deep supervision")

    return model


def UNet4Plus_wo_DeepSupv(num_classes, input_height, input_width, num_filters):
    """
    U-Net4+ without deep supervision
    """
    inputs = Input(shape=(input_height, input_width, 1))

    e1 = conv_block(inputs, num_filters)

    e2 = down_conv(e1, num_filters, 2)
    e2 = conv_block(e2, num_filters)

    e1_d_e3 = down_conv(e1, num_filters, 4)
    e2_d_e3 = down_conv(e2, num_filters, 2)
    e3 = Concatenate()([e1_d_e3, e2_d_e3])
    e3 = conv_block(e3, num_filters * 2)

    e1_d_e4 = down_conv(e1, num_filters, 8)
    e2_d_e4 = down_conv(e2, num_filters, 4)
    e3_d_e4 = down_conv(e3, num_filters, 2)
    e4 = Concatenate()([e1_d_e4, e2_d_e4, e3_d_e4])
    e4 = conv_block(e4, num_filters * 3)

    e1_d_e5 = down_conv(e1, num_filters, 16)
    e2_d_e5 = down_conv(e2, num_filters, 8)
    e3_d_e5 = down_conv(e3, num_filters, 4)
    e4_d_e5 = down_conv(e4, num_filters, 2)
    e5 = Concatenate()([e1_d_e5, e2_d_e5, e3_d_e5, e4_d_e5])
    e5 = conv_block(e5, num_filters * 4)

    e1_d_d4 = down_conv(e1, num_filters, 8)
    e2_d_d4 = down_conv(e2, num_filters, 4)
    e3_d_d4 = down_conv(e3, num_filters, 2)
    e4_d4 = conv(e4, num_filters)
    e5_u_d4 = up_conv(e5, num_filters, 2)
    d4 = Concatenate()([e1_d_d4, e2_d_d4, e3_d_d4, e4_d4, e5_u_d4])
    d4 = conv_block(d4, num_filters * 5)

    e1_d_d3 = down_conv(e1, num_filters, 4)
    e2_d_d3 = down_conv(e2, num_filters, 2)
    e3_d3 = conv(e3, num_filters)
    e4_u_d3 = up_conv(e4, num_filters, 2)
    e5_u_d3 = up_conv(e5, num_filters, 4)
    d4_u_d3 = up_conv(d4, num_filters, 2)
    d3 = Concatenate()([e1_d_d3, e2_d_d3, e3_d3, e4_u_d3, e5_u_d3, d4_u_d3])
    d3 = conv_block(d3, num_filters * 6)

    e1_d_d2 = down_conv(e1, num_filters, 2)
    e2_d2 = conv(e2, num_filters)
    e3_u_d2 = up_conv(e3, num_filters, 2)
    e4_u_d2 = up_conv(e4, num_filters, 4)
    e5_u_d2 = up_conv(e5, num_filters, 8)
    d4_u_d2 = up_conv(d4, num_filters, 4)
    d3_u_d2 = up_conv(d3, num_filters, 2)
    d2 = Concatenate()([e1_d_d2, e2_d2, e3_u_d2, e4_u_d2, e5_u_d2, d4_u_d2, d3_u_d2])
    d2 = conv_block(d2, num_filters * 7)

    e1_d1 = conv(e1, num_filters)
    e2_u_d1 = up_conv(e2, num_filters, 2)
    e3_u_d1 = up_conv(e3, num_filters, 4)
    e4_u_d1 = up_conv(e4, num_filters, 8)
    e5_u_d1 = up_conv(e5, num_filters, 16)
    d4_u_d1 = up_conv(d4, num_filters, 8)
    d3_u_d1 = up_conv(d3, num_filters, 4)
    d2_u_d1 = up_conv(d2, num_filters, 2)
    d1 = Concatenate()([e1_d1, e2_u_d1, e3_u_d1, e4_u_d1, e5_u_d1, d4_u_d1, d3_u_d1, d2_u_d1])
    d1 = conv_block(d1, num_filters * 8)

    outputs = Conv2D(num_classes, kernel_size=3, padding="same", activation="sigmoid", kernel_initializer="he_normal")(d1)

    model = Model(inputs=inputs, outputs=outputs, name="U-Net4+ without deep supervision")

    return model
