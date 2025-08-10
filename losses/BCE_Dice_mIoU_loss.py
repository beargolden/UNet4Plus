# coding=utf-8
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.losses import binary_crossentropy
from skimage.morphology import label


def iou_metric(y_true_in, y_pred_in, print_table=False):
    y_true = label(y_true_in > 0.5)
    y_pred = label(y_pred_in > 0.5)

    true_objects = len(np.unique(y_true))
    pred_objects = len(np.unique(y_pred))

    intersection = np.histogram2d(y_true.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(y_true, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # print('IOU {}'.format(iou))
    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")

    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)


def iou_metric_batch(y_true, y_pred):
    batch_size = y_true.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true[batch], y_pred[batch])
        metric.append(value)
    return np.array(np.mean(metric), dtype=np.float32)


def mean_iou_metric(y_true, y_pred):
    metric_value = tf.compat.v1.py_func(iou_metric_batch, [y_true, y_pred], tf.float32)
    return metric_value


def mean_iou_metric_loss(y_true, y_pred):
    loss = 1 - mean_iou_metric(y_true, y_pred)
    loss.set_shape((None,))
    return loss


def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2.0 * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def _BCE_Dice_mIoU_Loss(y_true, y_pred):
    """
    The segmentation loss is optimized as the weighted average of binary crossentropy,
    dice coefficient and mean intersection over union (IoU) which is evaluated with
    pixel accuracy, loss value and IoU. The IoU score calculation/implementation is
    as per the Kaggle Data Science Bowl Challenge 2018 (KDSB18), which is the more
    precise and accurate approach for computing IoU.

    :param y_true: the ground truth
    :param y_pred: the predicted
    :return: the weighted average loss value
    """
    loss = 0.4 * binary_crossentropy(y_true, y_pred) + \
           0.2 * dice_coef_loss(y_true, y_pred) + \
           0.4 * mean_iou_metric_loss(y_true, y_pred)
    return loss


def BCE_Dice_mIoU_Loss():
    return _BCE_Dice_mIoU_Loss
