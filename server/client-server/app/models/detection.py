from pydantic import BaseModel
from typing import List


class DetectionModel(BaseModel):
    class Detection(BaseModel):
        bbox: List[int]
        score: float
        category_id: int
        label: str
        segmentation: List[List[List[int]]]
    detections: List[Detection]
