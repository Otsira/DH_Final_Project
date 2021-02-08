from tensorflow.python.saved_model import tag_constants
import tensorflow as tf
# Image
from PIL import Image
import cv2
import numpy as np

from itertools import zip_longest
from typing import List
import random
import colorsys


class Yolo(object):
    config = None
    classes = None

    def __init__(self, config):
        self.config = config
        self.classes = list(self.read_class_names(
            self.config.CLASSES).values())
        self.model = tf.saved_model.load(
            self.config.PATH, tags=[tag_constants.SERVING])
        self.infer = self.model.signatures['serving_default']

    # makes predictions on a list of photos
    def predict(self, image_data):
        data = tf.constant(image_data)
        pred_bbox = self.infer(data)
        value = list(pred_bbox.values())[0]
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]
        return pred_conf, boxes

    def load_image(self, image: bytes, from_string: bool = True):
        if from_string:
            image = np.fromstring(image, np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data = [cv2.resize(
                image, (self.config.SIZE, self.config.SIZE)) / 255.]
        data = np.asarray(data).astype(np.float32)
        return data, image

    def load_images(self, images: List[bytes], from_string: bool = True):
        images_data = []
        images_cv2 = []
        for image in images:
            if from_string:
                image = np.fromstring(image, np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images_cv2.append(image)
            image_data = cv2.resize(
                image, (self.config.SIZE, self.config.SIZE)) / 255.
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)
        return images_data, images_cv2

    def draw_bbox(self, image, bboxes, classes=None, show_label=True, custom_labels=[]):
        classes = self.classes if not classes else classes
        num_classes = len(classes)
        image_h, image_w, _ = image.shape
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.)
                      for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)
        predictions = self.get_info(bboxes, classes)
        for pred, label in zip_longest(predictions, custom_labels, fillvalue=None):
            fontScale = 0.5
            if pred['class'] in classes:
                bbox_color = colors[pred['class_ind']]
                bbox_thick = int(0.6 * (image_h + image_w) / 600)
                coor = pred['coordinates']
                c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
                cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
                if show_label:
                    bbox_mess = '%s: %.2f' % (
                        pred['class'], pred['score'])
                    t_size = cv2.getTextSize(
                        bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                    c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
                    cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(
                        c3[1])), bbox_color, -1)  # filled
                    cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
                    if label:
                        cv2.putText(image, label, (c2[0], np.float32(c2[1] + 2)), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 255, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
            else:
                continue
        return image

    @classmethod
    def crop_objects(cls, img, bboxes, classes):
        predictions = cls.get_info(bboxes, classes)
        # create dictionary to hold count of objects for image name
        images = []
        for num, pred in enumerate(predictions):
            # get count of class for part of image name
            if pred['class'] in classes:
                # FIXME
                # counts[class_name] = counts.get(pred['class'], 0) + 1
                # get box coords
                xmin, ymin, xmax, ymax = pred['coordinates']
                # crop detection from image (take an additional 5 pixels around all edges)
                cropped_img = img[int(ymin)-5:int(ymax)+5,
                                  int(xmin)-5:int(xmax)+5]
                images.append({**pred, 'object_num': num, 'img': cropped_img})
            else:
                continue
        return images

    def process_pred(self, boxes, conf, shape):
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                conf, (tf.shape(conf)[0], -1, tf.shape(conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=self.config.IOU,
            score_threshold=self.config.SCORE
        )
        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = shape
        bboxes = self.format_boxes(boxes.numpy()[0], original_h, original_w)

        # hold all detection data in one variable
        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0],
                     valid_detections.numpy()[0]]
        return pred_bbox

    @staticmethod
    def get_info(pred, classes):
        results = []
        out_boxes, out_scores, out_classes, num_boxes = pred
        for i in range(num_boxes):
            if int(out_classes[i]) < 0 or int(out_classes[i]) > len(classes):
                continue
            coor = out_boxes[i]
            score = out_scores[i]
            class_ind = int(out_classes[i])
            class_name = classes[class_ind]
            results.append({'class': class_name, 'class_ind': class_ind,
                            'score': score, 'coordinates': coor})
        return results

    @staticmethod
    def read_class_names(class_file_name):
        names = {}
        with open(class_file_name, 'r') as data:
            for ID, name in enumerate(data):
                names[ID] = name.strip('\n')
        return names

    @staticmethod
    def format_boxes(bboxes, image_height, image_width):
        for box in bboxes:
            ymin = int(box[0] * image_height)
            xmin = int(box[1] * image_width)
            ymax = int(box[2] * image_height)
            xmax = int(box[3] * image_width)
            box[0], box[1], box[2], box[3] = xmin, ymin, xmax, ymax
        return bboxes
