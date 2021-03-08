from functools import lru_cache
from fastapi import FastAPI

from core.event_handlers import (start_app_handler,
                                 stop_app_handler)
from routes.api import api_router
from configs import configs
from core.logger import setup_logger_from_settings


@lru_cache()
def get_settings():
    return configs.Settings()


def init_app(settings: configs.Settings = get_settings()):

    app = FastAPI(
        title=settings.app_title,
        description=settings.app_description,
        version=settings.app_version,
        debug=settings.debug,
        docs_url='/'
    )
    app.include_router(api_router, prefix=settings.api_prefix)
    app.add_event_handler("startup", start_app_handler(app))
    app.add_event_handler("shutdown", stop_app_handler(app))

    setup_logger_from_settings()

    return app


app = init_app()
