from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from typing import Optional, List
from pydantic import BaseModel
from starlette import responses
from starlette.responses import Response, StreamingResponse
import json
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
async def read_plates(video: UploadFile = File(...), frames: Optional[int] = 5):
    car_tracker.create_tracker()
    # Sadly opencv only reads from disk
    with video.file as file:
        temp_video = NamedTemporaryFile('wb', delete=False)
        temp_video.write(file.read())
        file.close()
        # temp_video.close()
    vid = cv2.VideoCapture(temp_video.name)
    video = video_info(vid, frames)
    response = StreamingResponse(video)
    car_tracker.delete_tracker()
    return response


@router.post('/draw_plates')
async def draw_plates(video: UploadFile = File(...), frames: Optional[int] = 5):
    await car_tracker.create_tracker()
    # Sadly opencv only reads from disk
    with video.file as file:
        temp_video = NamedTemporaryFile('wb', delete=False)
        temp_video.write(file.read())
        file.close()
        # temp_video.close()
    vid = cv2.VideoCapture(temp_video.name)
    video = video_draw(vid, frames)
    response = StreamingResponse(
        video, media_type='media_type="image/jpg"')
    car_tracker.delete_tracker()
    return response


async def video_draw(vid, frames):
    await car_tracker.create_tracker()
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
            frame = car_tracker.draw_boxes(tracks, frame)
            data, img = plate_localizer.load_image(frame, False)
            conf, box = plate_localizer.predict(data)
            pred_bbox = plate_localizer.process_pred(box, conf, img.shape)
            plates = plate_localizer.crop_objects(
                img, pred_bbox, ['license_plate'])
            predictions = [plate_reader.read_plate(
                plate['img']) for plate in plates]
            license = [pred['plate'] for pred in predictions]
            frame = plate_localizer.draw_bbox(
                frame, pred_bbox, custom_labels=license)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, jpeg = cv2.imencode('.jpg', result)
        yield io.BytesIO(img.tobytes())
        # yield (b'--frame\r\n'
        #    b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        frame_num += 1


async def video_info(vid, frames):
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
            tracks = [{'id': track.track_id, 'class': track.get_class(
            ), 'coordinate': list(track.to_tlbr())} for track in tracks]
            predictions = [plate_reader.read_plate(
                plate['img']) for plate in plates]
            yield json.dumps({'frame': frame_num, 'tracks': tracks, 'licenses_plates': predictions})
        frame_num += 1
