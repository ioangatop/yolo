from starlette.responses import RedirectResponse
from functools import lru_cache
from fastapi import APIRouter
from fastapi import Depends

from configs import configs


@lru_cache()
def get_settings():
    return configs.Settings()


settings = get_settings()


router = APIRouter(
    tags=['utils']
)


@router.get('/docs')
@router.get('/')
async def docs():
    return RedirectResponse('/')


@router.get('/info', name='info')
async def info(settings: configs.Settings = Depends(get_settings)):
    return {
        'info': {
            'app_title': settings.app_title,
            'app_description': settings.app_description,
            'app_version': settings.app_version,
            'app_author': settings.app_author,
            'app_author_email': settings.app_author_email,
            'app_year': settings.app_year
        }
    }


@router.get('/settings', name='settings')
async def settings(app_settings: configs.Settings = Depends(get_settings)):
    return {
        'settings': {
            'triton_host': app_settings.triton_host,
            'triton_port': app_settings.triton_port,
            'triton_model_protocol': app_settings.triton_model_protocol,
            'label_mapping': app_settings.label_mapping,
            'reload': app_settings.reload,
            'debug': app_settings.debug,
        }
    }
