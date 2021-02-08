from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from typing import Optional, List
from pydantic import BaseModel
from starlette.responses import Response, StreamingResponse

from app.internal.license_plate.localizer import Localizer
from app.internal.license_plate.recognizer import Recognizer
from app.internal.tracker.tracker import Tracker

import io
import cv2
import numpy as np
from PIL import Image
from tempfile import NamedTemporaryFile

from shapely.geometry import Point, point
from shapely.geometry.polygon import Polygon

car_tracker = Tracker()
plate_localizer = Localizer()
plate_reader = Recognizer()


router = APIRouter(
    prefix="/videos",
    tags=["videos"],
    responses={404: {"description": "Not found"}},
)


@router.post('/read_plates')
def read_plates(video: UploadFile = File(...), frames: Optional[int] = 5):
    car_tracker.create_tracker()
    # Sadly opencv only reads from disk
    with video.file as file:
        temp_video = NamedTemporaryFile('wb', delete=False)
        temp_video.write(file.read())
        file.close()
        # temp_video.close()
    vid = cv2.VideoCapture(temp_video.name)
    vid_len = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = 0
    while True:
        return_value, frame = vid.read()
        if return_value:
            data, img = car_tracker.load_image(frame, False)
        else:
            break
        if frame_num % frames == 0:
            conf_cars, box_cars = car_tracker.predict(data)
            pred_bbox_cars = car_tracker.process_pred(
                box_cars, conf_cars, img.shape)
            car_detect = car_tracker.process_tracker(pred_bbox_cars, img)
            tracks = car_tracker.predict_tracker(car_detect)
            data, img = plate_localizer.load_image(frame, False)
            conf, box = plate_localizer.predict(data)
            pred_bbox = plate_localizer.process_pred(box, conf, img.shape)
            plates = plate_localizer.crop_objects(
                img, pred_bbox, ['license_plate'])
            track_plates = []
            for track in tracks:
                xmin, ymin, xmax, ymax = track.to_tlbr()
                for plate in plates:
                    pxmin, pymin, pxmax, pymax = plate['coordinates']
                    if (xmax < pxmax and ymax > pymax):
                        track_plates.append(
                            {'track': track.track_id, 'class': track.class_name,  **plate})

            frame_num += 1

    print(vid_len)
    return {'name': vid_len}
