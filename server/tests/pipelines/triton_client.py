from tritonclient.utils import InferenceServerException
import tritonclient.http as tritonhttpclient
import sys

from .preprocess import ImageLoader
from .postprocess import ProcessDetections



class TRTInferenceModule:
    def __init__(self, host='0.0.0.0', port='8000', model_name='detector', model_version='1'):
        self.model_name = model_name
        self.model_version = str(model_version)
        self.triton_client = self.setup_triton_client(f'{host}:{port}')

        self.model_config = self.model_config()
        self.model_metadata = self.model_metadata()

        self.input_names = self._get_input_names()
        self.output_names = self._get_output_names()
        self.max_batch_size = self.max_batch_size()


    @staticmethod
    def setup_triton_client(url):
        triton_client = tritonhttpclient.InferenceServerClient(url=url)
        return triton_client


    def _load_model(self, model_name=None):
        if model_name is None:
            model_name = self.model_name
        self.triton_client.load_model(model_name)


    def _unload_model(self, model_name=None):
        if model_name is None:
            model_name = self.model_name
        self.triton_client.load_model(model_name)


    def _health(self):
        if not self.triton_client.is_server_live():
            print("FAILED : is_server_live")
            sys.exit(1)

        if not self.triton_client.is_server_ready():
            print("FAILED : is_server_ready")
            sys.exit(1)

        if not self.triton_client.is_model_ready(self.model_name):
            print("FAILED : is_model_ready")
            sys.exit(1)


    def model_config(self):
        try:
            model_config = self.triton_client.get_model_config(
                model_name=self.model_name, model_version=self.model_version)
        except InferenceServerException as e:
            print("failed to retrieve the config: " + str(e))
            sys.exit(1)
        return model_config


    def model_metadata(self):
        try:
            model_metadata = self.triton_client.get_model_metadata(
                model_name=self.model_name, model_version=self.model_version)
        except InferenceServerException as e:
            print("Failed to retrieve the metadata: " + str(e))
            sys.exit(1)
        return model_metadata


    def max_batch_size(self):
        max_batch_size = 0
        if 'max_batch_size' in self.model_config:
            max_batch_size = self.model_config['max_batch_size']
        return max_batch_size


    def _get_input_names(self):
        input_names = [input['name'] for input in self.model_config['input']]
        return input_names


    def _get_output_names(self):
        output_names = [input['name'] for input in self.model_config['output']]
        return output_names


    def _to_tt_inputs(self, inputs):
        tt_inputs = []
        for input, conf in zip([inputs], self.model_metadata['inputs']):
            tt_input = tritonhttpclient.InferInput(
                conf['name'],
                list(input.shape),
                conf['datatype']
            )
            tt_input.set_data_from_numpy(input)
            tt_inputs.append(tt_input)
        return tt_inputs


    def infer(self, inputs):
        tt_inputs = self._to_tt_inputs(inputs)
        response = self.triton_client.infer(self.model_name, tt_inputs)
        return response


    def __call__(self, inputs):
        return self.infer(inputs)
