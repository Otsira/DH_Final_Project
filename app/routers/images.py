from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from typing import Optional, List
from pydantic import BaseModel
from starlette.responses import Response, StreamingResponse


from app.internal.license_plate.localizer import Localizer
from app.internal.license_plate.recognizer import Recognizer

import io
import cv2
import numpy as np

plate_localizer = Localizer()
plate_reader = Recognizer()

router = APIRouter(
    prefix="/images",
    tags=["images"],
    responses={404: {"description": "Not found"}},
)


@router.post('/read_plates')
async def read_plates(image: bytes = File(...)):
    data, img = plate_localizer.load_image(image)
    conf, box = plate_localizer.predict(data)
    pred_bbox = plate_localizer.process_pred(box, conf, img.shape)
    plates = plate_localizer.crop_objects(img, pred_bbox, ['license_plate'])
    predictions = [plate_reader.read_plate(plate['img']) for plate in plates]
    return predictions


@router.post('/draw_plates')
async def draw_plates(image: bytes = File(...)):
    data, img = plate_localizer.load_image(image)
    conf, box = plate_localizer.predict(data)
    pred_bbox = plate_localizer.process_pred(box, conf, img.shape)
    plates = plate_localizer.crop_objects(img, pred_bbox, ['license_plate'])
    predictions = [plate_reader.read_plate(plate['img']) for plate in plates]
    license = [pred['plate'] for pred in predictions]
    img = plate_localizer.draw_bbox(img, pred_bbox, custom_labels=license)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    _, img = cv2.imencode('.png', img)
    return StreamingResponse(io.BytesIO(img.tobytes()), media_type="image/png")
