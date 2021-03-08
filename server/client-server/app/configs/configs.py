from pydantic import BaseSettings


class Settings(BaseSettings):
    # app info
    app_title: str = 'Triton Inference Server Middleware'
    app_description: str = 'Triton I/O (Middleware) Client Server bridges \
        the souce and the model server, shaping the data to the \
        expected format for the triton model.'
    app_version: str = '1.2'
    app_author: str = 'BrainCreators'
    app_author_email: str = 'info@braincreatos.com'
    app_year: str = '2021'

    # api settings
    api_prefix: str = '/api'

    # triton settings
    triton_host: str = '0.0.0.0'
    triton_port: str = '8001'
    triton_model_protocol: str = 'gRPC'

    # on_launch
    debug: bool = False
    reload: bool = False

    # read settings from .env
    class Config:
        env_file = '.env'


settings = Settings()
