from easydict import EasyDict as edict
from tensorflow.python.keras.activations import softmax
from tensorflow.keras import backend as K
import tensorflow as tf


__C = edict()
# Consumers can get config by: from config import cfg

detector = __C

# YOLO options
# __C.YOLO                      = edict()
__C.IOU = 0.45
__C.SCORE = 0.50
__C.SIZE = 416
__C.PATH = './internal/models/license_detector'
__C.CLASSES = "./internal/models/license_detector/custom.names"
__C.ANCHORS = [12, 16, 19, 36, 40, 28, 36, 75,
               76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
__C.ANCHORS_V3 = [10, 13, 16, 30, 33, 23, 30, 61,
                  62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
__C.ANCHORS_TINY = [23, 27, 37, 58, 81, 82, 81, 82, 135, 169, 344, 319]
__C.STRIDES = [8, 16, 32]
__C.STRIDES_TINY = [16, 32]
__C.XYSCALE = [1.2, 1.1, 1.05]
__C.XYSCALE_TINY = [1.05, 1.05]
__C.ANCHOR_PER_SCALE = 3
__C.IOU_LOSS_THRESH = 0.5

recognizer = edict()
recognizer.path = './internal/models/license_plate_ocr.h5'


def cat_acc(y_true, y_pred):
    y_true = K.reshape(y_true, shape=(-1, 7, 37))
    y_pred = K.reshape(y_pred, shape=(-1, 7, 37))
    return K.mean(tf.keras.metrics.categorical_accuracy(y_true, y_pred))


def plate_acc(y_true, y_pred):
    '''
    How many plates were correctly classified
    If Ground Truth is ABC 123
    Then prediction ABC 123 would score 1
    else ABD 123 would score 0
    Avg these results (1 + 0) / 2 -> Gives .5 accuracy
    (Half of the plates were completely corrected classified)
    '''
    y_true = K.reshape(y_true, shape=(-1, 7, 37))
    y_pred = K.reshape(y_pred, shape=(-1, 7, 37))
    et = K.equal(K.argmax(y_true), K.argmax(y_pred))
    return K.mean(
        K.cast(K.all(et, axis=-1, keepdims=False), dtype='float32')
    )


def top_3_k(y_true, y_pred):
    # Reshape into 2-d
    y_true = K.reshape(y_true, (-1, 37))
    y_pred = K.reshape(y_pred, (-1, 37))
    return K.mean(
        tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)
    )

# Custom loss


def cce(y_true, y_pred):
    y_true = K.reshape(y_true, shape=(-1, 37))
    y_pred = K.reshape(y_pred, shape=(-1, 37))

    return K.mean(
        tf.keras.losses.categorical_crossentropy(
            y_true, y_pred, from_logits=False, label_smoothing=0.2
        )
    )


custom_objects = {
    'cce': cce,
    'cat_acc': cat_acc,
    'plate_acc': plate_acc,
    'top_3_k': top_3_k,
    'softmax': softmax
}
