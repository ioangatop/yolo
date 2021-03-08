from tritonclient.utils import InferenceServerException
from functools import partial

import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
import asyncio

from .postprocess import ProcessDetections
from .preprocess import ImageLoader
from .database import AsyncDatabase

from core.messages import (INFERENCE_ERROR,
                           FAILED_TO_CONNECT_TO_TRITON_SERVER,
                           UNKNOWN_PROTOCOL)



class TRTInferenceModule:
    def __init__(
        self,
        host='192.168.1.113',
        port='8001',
        protocol='gRPC'
    ):

        self.protocol = self.setup_protocol(protocol)
        self.trt_client = self.setup_trt_client(f'{host}:{port}')

        self.metadata = self.load_metadata()
        self.output_names = self.output_names()

        self.db = AsyncDatabase()
        self.dataloader = ImageLoader()
        self.postprocess = ProcessDetections()


    @staticmethod
    def setup_protocol(protocol):
        protocol = protocol.lower()
        if protocol not in ['http', 'grpc']:
            raise ValueError(UNKNOWN_PROTOCOL.format(protocol))
        return protocol


    @staticmethod
    def async_callback(db, postprocess, result, error):
        if error is not None:
            raise ValueError(INFERENCE_ERROR.format(error))
        db.put(postprocess(result))


    def setup_trt_client(self, url):
        def check_connection(trt_client):
            try:
                if (trt_client.is_server_live()
                    and trt_client.is_server_ready()):
                    return
            except:
                raise ValueError(FAILED_TO_CONNECT_TO_TRITON_SERVER.format(url, self.protocol))

        if self.protocol == 'http':
            trt_client = httpclient.InferenceServerClient(
                url=url, verbose=False)
        elif self.protocol == 'grpc':
            trt_client = grpcclient.InferenceServerClient(
                url=url, verbose=False)

        check_connection(trt_client)
        return trt_client


    def info(self):
        return self.trt_client.get_model_repository_index(as_json=True)


    def stats(self, model_name='', model_version=''):
        return self.trt_client.get_inference_statistics(model_name,
                                                        model_version,
                                                        as_json=True)


    def health(self, model_name=None, model_version=''):
        if model_name is None:
            return {
                'heath': {
                    'is_server_live': self.trt_client.is_server_live(),
                    'is_server_ready': self.trt_client.is_server_ready(),
                }
            }
        else:
            return {
                'heath': {
                    'is_model_ready': self.trt_client.is_model_ready(model_name,
                                                                     model_version)
                }
            }


    def _load_model(self, model_name):
        self.trt_client.load_model(model_name)


    def _unload_model(self, model_name):
        self.trt_client.load_model(model_name)


    def load_config(self):
        model_info = [[model.get('name'), model.get('version') or '1'] for model in self.info()['models']]
        config = {model[0]: {model[1]: self.model_config(*model)} for model in model_info}
        return config


    def load_metadata(self):
        model_info = [[model.get('name'), model.get('version') or '1'] for model in self.info()['models']]
        metadata = {model[0]: {model[1]: self.model_metadata(*model)} for model in model_info}
        return metadata


    def model_config(self, model_name, model_version=''):
        model_config = self.trt_client.get_model_config(model_name, model_version)
        return model_config


    def model_metadata(self, model_name, model_version=''):
        model_metadata = self.trt_client.get_model_metadata(model_name, model_version)
        return model_metadata


    def input_names(self):
        model_info = [[model['name'], model['version']] for model in self.info()['models']]
        input_names = {model[0]: {model[1]: self.model_input_names(*model)} for model in model_info}
        return input_names


    def output_names(self):
        model_info = [[model['name'], model['version']] for model in self.info()['models']]
        output_names = {model[0]: {model[1]: self.model_output_names(*model)} for model in model_info}
        return output_names


    def model_input_names(self, model_name, model_version='1'):
        if self.protocol == 'http':
            input_names = [input['name'] for input in self.metadata[model_name][model_version]['inputs']]
        elif self.protocol == 'grpc':
            input_names = [input.name for input in self.metadata[model_name][model_version].inputs]
        return input_names


    def model_output_names(self, model_name, model_version='1'):
        if self.protocol == 'http':
            output_names = [output['name'] for output in self.metadata[model_name][model_version]['outputs']]
        elif self.protocol == 'grpc':
            output_names = [output.name for output in self.metadata[model_name][model_version].outputs]
        return output_names


    def _to_trt(self, inputs, model_name, model_version='1'):
        tt_inputs = []
        if self.protocol == 'http':
            input_metadata = self.metadata[model_name][model_version]['inputs']
            for input, metadata in zip([inputs], input_metadata):
                tt_input = httpclient.InferInput(
                    metadata['name'],
                    list(input.shape),
                    metadata['datatype']
                )
                tt_input.set_data_from_numpy(input)
                tt_inputs.append(tt_input)

        elif self.protocol == 'grpc':
            input_metadata = self.metadata[model_name][model_version].inputs
            for input, metadata in zip([inputs], input_metadata):
                tt_input = grpcclient.InferInput(
                    metadata.name,
                    list(input.shape),
                    metadata.datatype
                )
                tt_input.set_data_from_numpy(input)
                tt_inputs.append(tt_input)

        return tt_inputs


    def infer(self, inputs, model_name, model_version='1'):
        tt_inputs = self._to_trt(inputs, model_name, model_version)
        response = self.trt_client.infer(model_name, tt_inputs, model_version)
        return response


    def detect(self, input, model_name, model_version='1', **kwargs):
        inputs = self.dataloader(input)
        response = self.infer(inputs, model_name, model_version)
        detections = self.postprocess(response, trt_output_names=self.output_names[model_name][model_version], **kwargs)
        return detections


    def get_async_result(self):
        return self.db.get()


    def async_infer(self, inputs, model_name, model_version='1', **kwargs):
        tt_inputs = self._to_trt(inputs, model_name, model_version)
        self.trt_client.async_infer(model_name, tt_inputs,
                                       partial(self.async_callback, self.db,
                                               partial(self.postprocess,
                                                       trt_output_names=self.output_names[model_name][model_version],
                                                       **kwargs)),
                                        model_version=model_version,
                                        client_timeout=None)


    async def async_detect(self, input, model_name, model_version='1', **kwargs):
        inputs = self.dataloader(input)
        self.async_infer(inputs, model_name, model_version, **kwargs)
        await asyncio.sleep(0.1)
        detections = self.get_async_result()
        return detections


    def __call__(self, input, model_name, model_version='1', **kwargs):
        return self.detect(input, model_name, model_version, **kwargs)
