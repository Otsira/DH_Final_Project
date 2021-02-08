from os import name
from app.internal.yolo import Yolo
from .config import detector, sorter
# deep sort imports
from .deep_sort import preprocessing, nn_matching
from .deep_sort.detection import Detection
from .deep_sort.tracker import Tracker as deepTrack
from .tools import generate_detections as gdet

import matplotlib.pyplot as plt
import numpy as np
import cv2


class Tracker(Yolo):
    def __init__(self):
        self.allowed_classes = ["car"]
        super().__init__(detector)
        self.encoder = gdet.create_box_encoder(sorter.path, batch_size=1)
        self.metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", sorter.max_cosine_distance, sorter.nn_budget)
        cmap = plt.get_cmap('tab20b')
        self.coco_names = list(self.read_class_names(
            self.config.COCO).values())
        self.colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

    def create_tracker(self):
        self.deepTracker = deepTrack(self.metric)

    def delete_tracker(self):
        del self.deepTracker

    def process_tracker(self, pred, img):
        boxes, scores, names, _ = self.remove_unallowed(pred)
        features = self.encoder(img,  boxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox,
                      score, class_name, feature in zip(boxes, scores, names, features)]
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(
            boxs, classes, sorter.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        return detections

    def predict_tracker(self, detections):
        self.deepTracker.predict()
        self.deepTracker.update(detections)
        return self.deepTracker.tracks

    def remove_unallowed(self, pred):
        box, scores, classes, num_obj = pred
        names = []
        deleted_indx = []
        for i in range(num_obj):
            class_indx = int(classes[i])
            class_name = self.coco_names[class_indx]
            if class_name not in self.allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        box = np.delete(box, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)
        return box, scores, names, num_obj

    def draw_boxes(self, tracks, frame):
        for track in tracks.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            # draw bbox on screen
            color = self.colors[int(track.track_id) % len(self.colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(
                bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(
                len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),
                        (int(bbox[0]), int(bbox[1]-10)), 0, 0.75, (255, 255, 255), 2)
        return frame
