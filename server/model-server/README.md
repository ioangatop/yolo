# Triton Model Server

> The __Triton Model Server__ is responsible for the model hosting.


### Setup

#### **Docker Image**

To `build` the docker image, please move to the parent folder of this reposetory and execute the following command:

```
docker-compose -f docker/docker-compose.yml build triton-model-server
```

The blueprint of the docker-image can be found at `docker/Dockerfile.triton_model_server`. The image is also available on our docker hubs:

- Internal hub: `hub.hq.braincreators.com/brainmatter/detectron2_triton_model_server:1.0`

- External hub: `hub.braincreators.com/detectron2_triton_model_server:1.0`

<br />


#### **Model Import**


The model repository (i.e. `models`) should have the following structure:

```
models
 ├─ dali
 │   ├─ config.pbtxt
 │   └─ 1
 │      └─ model.dali
 │
 ├─ model
 │   ├─ config.pbtxt
 │   └─ 1
 │      └─ model.pt
 │
 └─ detector
    ├─ config.pbtxt
    └─ 1
       └─ <empty-folder>
```

Specifically:

- `dali` is responsible for the image pre-processing. To compile and to export the pipeline `model.dali`, please use the script `tools/export/export_dali.py`.

- `model` is a serialized (*TorchScript*) version of a pytorch model. To export your model in this format, please use the script `tools/export/export_model.py`.

- `detector` is a *ensemble* triton-platform, which orchestrates the inference process.

<br />


### Run

To `run` the server via docker-compose, please first export the model path to the `TRT_MODEL_DIR` enviroment flag and execute the following command:

```
docker-compose -f docker/docker-compose.yml up triton-model-server
```

