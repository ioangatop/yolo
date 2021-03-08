from fastapi.responses import FileResponse
from functools import lru_cache
from pydantic import HttpUrl
from fastapi import APIRouter, Query
from typing import Optional

from services import TRTInferenceModule, Visualizer
from configs import configs
from models import DetectionModel


@lru_cache()
def get_settings():
    return configs.Settings()


settings = get_settings()
trt_client = TRTInferenceModule(settings.triton_host,
                                settings.triton_port,
                                settings.triton_model_protocol)


router = APIRouter(
    prefix='/models',
    tags=['models']
)


@router.get('/info', name="server info")
@router.get('/', name="server info")
async def server_info():
    return trt_client.info()


@router.get('/stats', name="server stats")
async def server_stats():
    return trt_client.stats()


@router.get('/health', name="server health")
async def server_health():
    return trt_client.health()


@router.get('/{model_name}/{model_version}/health', name='model health')
@router.get('/{model_name}/health', name='model health')
async def model_health(
    model_name: str = Query('detector', description='Model Name'),
    model_version: Optional[str] = Query('1', description='Model Version')
):
    return trt_client.health(model_name, model_version)


@router.get('/{model_name}/{model_version}/infer', response_model=DetectionModel, name="inference")
@router.get('/{model_name}/infer', response_model=DetectionModel, name="inference")
@router.get('/infer', response_model=DetectionModel, name="inference")
async def infer(
    model_name: Optional[str] = Query('detector', description='Model Name'),
    model_version: Optional[str] = Query('1', description='Model Version'),
    url : HttpUrl = Query('https://farm4.staticflickr.com/3491/4023151748_cb042e1794_z.jpg', description='Image URL'),
    threshold : Optional[float] = Query(0.3, description='Confidence score threshold', ge=0.0, le=1.0),
    iou_threshold : Optional[float] = Query(0.5, description='IOU detection threshold', ge=0.0, le=1.0),
    class_id : Optional[int] = Query(None, description='Class id filtering', ge=0),
) -> DetectionModel:
    """
    Model Inference API. 

    - **url**: Image URL
    - **threshold**: Confidence score threshold
    - **iou_threshold**: IOU detection threshold
    - **class_id**: Class id filtering
    \f
    :param item: User input.
    """
    detections = await trt_client.async_detect(url, model_name, model_version,
                                               threshold=threshold,
                                               iou_threshold=iou_threshold,
                                               class_id=class_id)
    return {'detections': detections}



@router.get('/{model_name}/{model_version}/infer_and_draw', name='inference and draw')
@router.get('/{model_name}/infer_and_draw', name='inference and draw')
@router.get('/infer_and_draw', name='inference and draw')
async def infer_and_draw(
    model_name: Optional[str] = Query('detector', description='Model Name'),
    model_version: Optional[str] = Query('1', description='Model Version'),
    url : HttpUrl = Query('https://farm4.staticflickr.com/3491/4023151748_cb042e1794_z.jpg', description='Image URL'),
    threshold : Optional[float] = Query(0.3, description='Confidence score threshold', ge=0.0, le=1.0),
    iou_threshold : Optional[float] = Query(0.5, description='IOU detection threshold', ge=0.0, le=1.0),
    class_id : Optional[int] = Query(None, description='Class id filtering', ge=0),
) -> FileResponse:
    """
    Model Inference API that draws the predictions on the given image. 

    - **url**: Image URL
    - **threshold**: Confidence score threshold
    - **iou_threshold**: IOU detection threshold
    - **class_id**: Class id filtering
    \f
    :param item: User input.
    """
    visualizer = Visualizer()
    detections = trt_client.detect(url, model_name, model_version,
                                   threshold=threshold,
                                   iou_threshold=iou_threshold,
                                   class_id=class_id)
    image = visualizer(url, detections, exclude=['labels', 'scores'], save_as='./tmp/image.jpg')
    return FileResponse('./tmp/image.jpg')
