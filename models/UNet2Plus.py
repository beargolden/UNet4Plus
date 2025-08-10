# coding=utf-8
from keras.layers import *
from keras.models import *


def conv_block(input_tensor, num_filters):
    out = Conv2D(num_filters, kernel_size=3, padding="same", activation="relu",
                 kernel_initializer="he_normal")(input_tensor)
    out = Conv2D(num_filters, kernel_size=3, padding="same", activation="relu",
                 kernel_initializer="he_normal")(out)
    return out


def down_maxpool(input_tensor):
    out = MaxPooling2D(pool_size=2)(input_tensor)
    return out


def up_conv(input_tensor, num_filters):
    out = UpSampling2D(size=2, interpolation="bilinear")(input_tensor)
    out = Conv2D(num_filters, kernel_size=3, padding="same", activation="relu",
                 kernel_initializer="he_normal")(out)
    return out


def UNet2Plus_w_DeepSupv(num_classes, input_height, input_width, num_filters):
    """
    U-Net++ with deep supervision
    Paper : https://arxiv.org/abs/1807.10165
    """
    inputs = Input(shape=(input_height, input_width, 1))

    filters = [num_filters, num_filters * 2, num_filters * 4, num_filters * 8, num_filters * 16]

    x0_0 = conv_block(inputs, filters[0])

    x1_0 = down_maxpool(x0_0)
    x1_0 = conv_block(x1_0, filters[1])

    x0_1 = up_conv(x1_0, filters[0])
    x0_1 = Concatenate()([x0_0, x0_1])
    x0_1 = conv_block(x0_1, filters[0])

    x2_0 = down_maxpool(x1_0)
    x2_0 = conv_block(x2_0, filters[2])

    x1_1 = up_conv(x2_0, filters[1])
    x1_1 = Concatenate()([x1_0, x1_1])
    x1_1 = conv_block(x1_1, filters[1])

    x0_2 = up_conv(x1_1, filters[0])
    x0_2 = Concatenate()([x0_0, x0_1, x0_2])
    x0_2 = conv_block(x0_2, filters[0])

    x3_0 = down_maxpool(x2_0)
    x3_0 = conv_block(x3_0, filters[3])

    x2_1 = up_conv(x3_0, filters[2])
    x2_1 = Concatenate()([x2_0, x2_1])
    x2_1 = conv_block(x2_1, filters[2])

    x1_2 = up_conv(x2_1, filters[1])
    x1_2 = Concatenate()([x1_0, x1_1, x1_2])
    x1_2 = conv_block(x1_2, filters[1])

    x0_3 = up_conv(x1_2, filters[0])
    x0_3 = Concatenate()([x0_0, x0_1, x0_2, x0_3])
    x0_3 = conv_block(x0_3, filters[0])

    x4_0 = down_maxpool(x3_0)
    x4_0 = conv_block(x4_0, filters[4])

    x3_1 = up_conv(x4_0, filters[3])
    x3_1 = Concatenate()([x3_0, x3_1])
    x3_1 = conv_block(x3_1, filters[3])

    x2_2 = up_conv(x3_1, filters[2])
    x2_2 = Concatenate()([x2_0, x2_1, x2_2])
    x2_2 = conv_block(x2_2, filters[2])

    x1_3 = up_conv(x2_2, filters[1])
    x1_3 = Concatenate()([x1_0, x1_1, x1_2, x1_3])
    x1_3 = conv_block(x1_3, filters[1])

    x0_4 = up_conv(x1_3, filters[0])
    x0_4 = Concatenate()([x0_0, x0_1, x0_2, x0_3, x0_4])
    x0_4 = conv_block(x0_4, filters[0])

    outputs = Average()([x0_1, x0_2, x0_3, x0_4])
    outputs = Conv2D(num_classes, kernel_size=3, padding="same", activation="sigmoid",
                     kernel_initializer="he_normal")(outputs)

    model = Model(inputs=inputs, outputs=outputs, name="U-Net++ with deep supervision")

    return model


def UNet2Plus_wo_DeepSupv(num_classes, input_height, input_width, num_filters):
    """
    U-Net++ without deep supervision
    Paper : https://arxiv.org/abs/1807.10165
    """
    inputs = Input(shape=(input_height, input_width, 1))

    filters = [num_filters, num_filters * 2, num_filters * 4, num_filters * 8, num_filters * 16]

    x0_0 = conv_block(inputs, filters[0])

    x1_0 = down_maxpool(x0_0)
    x1_0 = conv_block(x1_0, filters[1])

    x0_1 = up_conv(x1_0, filters[0])
    x0_1 = Concatenate()([x0_0, x0_1])
    x0_1 = conv_block(x0_1, filters[0])

    x2_0 = down_maxpool(x1_0)
    x2_0 = conv_block(x2_0, filters[2])

    x1_1 = up_conv(x2_0, filters[1])
    x1_1 = Concatenate()([x1_0, x1_1])
    x1_1 = conv_block(x1_1, filters[1])

    x0_2 = up_conv(x1_1, filters[0])
    x0_2 = Concatenate()([x0_0, x0_1, x0_2])
    x0_2 = conv_block(x0_2, filters[0])

    x3_0 = down_maxpool(x2_0)
    x3_0 = conv_block(x3_0, filters[3])

    x2_1 = up_conv(x3_0, filters[2])
    x2_1 = Concatenate()([x2_0, x2_1])
    x2_1 = conv_block(x2_1, filters[2])

    x1_2 = up_conv(x2_1, filters[1])
    x1_2 = Concatenate()([x1_0, x1_1, x1_2])
    x1_2 = conv_block(x1_2, filters[1])

    x0_3 = up_conv(x1_2, filters[0])
    x0_3 = Concatenate()([x0_0, x0_1, x0_2, x0_3])
    x0_3 = conv_block(x0_3, filters[0])

    x4_0 = down_maxpool(x3_0)
    x4_0 = conv_block(x4_0, filters[4])

    x3_1 = up_conv(x4_0, filters[3])
    x3_1 = Concatenate()([x3_0, x3_1])
    x3_1 = conv_block(x3_1, filters[3])

    x2_2 = up_conv(x3_1, filters[2])
    x2_2 = Concatenate()([x2_0, x2_1, x2_2])
    x2_2 = conv_block(x2_2, filters[2])

    x1_3 = up_conv(x2_2, filters[1])
    x1_3 = Concatenate()([x1_0, x1_1, x1_2, x1_3])
    x1_3 = conv_block(x1_3, filters[1])

    x0_4 = up_conv(x1_3, filters[0])
    x0_4 = Concatenate()([x0_0, x0_1, x0_2, x0_3, x0_4])
    x0_4 = conv_block(x0_4, filters[0])

    outputs = Conv2D(num_classes, kernel_size=3, padding="same", activation="sigmoid",
                     kernel_initializer="he_normal")(x0_4)

    model = Model(inputs=inputs, outputs=outputs, name="U-Net++ without deep supervision")

    return model
