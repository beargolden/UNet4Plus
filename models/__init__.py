from __future__ import absolute_import

from .UNet import *
from .UNet1Plus import *
from .UNet2Plus import *
from .UNet3Plus import *
from .UNet4Plus import *

MODEL_FACTORY = {
    "unet": UNet,
    "unet1plus_w_deepsupv": UNet1Plus_w_DeepSupv,
    "unet1plus_wo_deepsupv": UNet1Plus_wo_DeepSupv,
    "unet2plus_w_deepsupv": UNet2Plus_w_DeepSupv,
    "unet2plus_wo_deepsupv": UNet2Plus_wo_DeepSupv,
    "unet3plus_w_deepsupv": UNet3Plus_w_DeepSupv,
    "unet3plus_wo_deepsupv": UNet3Plus_wo_DeepSupv,
    "unet4plus_w_deepsupv": UNet4Plus_w_DeepSupv,
    "unet4plus_wo_deepsupv": UNet4Plus_wo_DeepSupv,
}


def show_avai_models():
    """Displays available models.
    Examples::
        >>> from models import *
        >>> models.show_avai_models()
    """
    print(list(MODEL_FACTORY.keys()))


def build_model(name, num_classes, input_height, input_width, num_filters):
    avai_models = list(MODEL_FACTORY.keys())
    if name not in avai_models:
        raise KeyError("Unknown model: {}. Must be one of {}".format(name, avai_models))
    return MODEL_FACTORY[name](num_classes,
                               input_height=input_height,
                               input_width=input_width,
                               num_filters=num_filters)
