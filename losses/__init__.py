from __future__ import absolute_import

from losses.BCE_Dice_loss import BCE_Dice_Loss
from losses.BCE_Dice_mIoU_loss import BCE_Dice_mIoU_Loss

LOSS_FACTORY = {
    "bce_dice": BCE_Dice_Loss(),
    "bce_dice_miou": BCE_Dice_mIoU_Loss(),
}


def show_avai_losses():
    """Displays available losses.
    Examples::
        >>> from losses import *
        >>> losses.show_avai_losses()
    """
    print(list(LOSS_FACTORY.keys()))


def build_loss(name):
    avai_losses = list(LOSS_FACTORY.keys())
    if name not in avai_losses:
        raise KeyError("Unknown loss: {}. Must be one of {}".format(name, avai_losses))
    return LOSS_FACTORY[name]
