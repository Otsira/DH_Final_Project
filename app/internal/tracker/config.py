from easydict import EasyDict as edict


__C = edict()
# Consumers can get config by: from config import cfg

detector = __C

# YOLO options
# __C.YOLO                      = edict()
__C.IOU = 0.45
__C.SCORE = 0.50
__C.SIZE = 416
__C.PATH = './app/internal/models/car_yolo'
__C.CLASSES = "./app/internal/models/car_yolo/car.names"
__C.COCO = './app/internal/models/car_yolo/coco.names'
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

sorter = edict()

sorter.max_cosine_distance = 0.4
sorter.nn_budget = None
sorter.nms_max_overlap = 1.0
sorter.path = './app/internal/models/deep_sort.pb'
