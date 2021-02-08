from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from typing import Optional, List
from pydantic import BaseModel
from starlette.responses import Response, StreamingResponse

from internal.license_plate.localizer import Localizer
from internal.license_plate.recognizer import Recognizer
from internal.tracker.tracker import Tracker

import io
import cv2
import numpy as np

car_tracker = Tracker()
plate_localizer = Localizer()
plate_reader = Recognizer()

router = APIRouter(
    prefix="/videos",
    tags=["videos"],
    responses={404: {"description": "Not found"}},
)


@router.post('/read_plates')
async def read_plates(video: bytes = UploadFile(...)):
    car_tracker.create_tracker()
