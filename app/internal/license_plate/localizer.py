from app.internal.yolo import Yolo
from .config import detector


class Localizer(Yolo):
    classes = ['license_plate']

    def __init__(self):
        super().__init__(detector)
