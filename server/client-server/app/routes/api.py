from fastapi import APIRouter

from .models import router as model_router
from .utils import router as utils_router


api_router = APIRouter()
api_router.include_router(model_router)
api_router.include_router(utils_router)
