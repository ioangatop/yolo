# Triton Client Server

> __Triton I/O__ _(Middleware)_ __Client Server__ bridges the souce and the model server, shaping the data to the expected format for the triton model.

### Setup

To `build` the docker image, please move to the parent folder of this reposetory and execute the following command:

```
docker-compose -f docker/docker-compose.yml build triton-client-server
```

The blueprint of the docker-image can be found at `docker/Dockerfile.triton_client_server`. The image is also available on our docker hubs:

- Internal hub: `hub.hq.braincreators.com/brainmatter/detectron2_triton_client_server:1.2`

- External hub: `hub.braincreators.com/detectron2_triton_client_server:1.2

<br />

### Run

To `run` the server via docker-compose, execute the following command:

```
docker-compose -f docker/docker-compose.yml up triton-client-server
```

The `docker-compose` will launch the file `server/app.py`, which expects the following arguments:

- `HOST`: API host
- `PORT`: API port
- `mode`: Serve mode ("debug", "develop" or "production")
- `TRITON_HOST`: Triton API host
- `TRITON_PORT`: Triton API port
- `TRITON_MODEL_PROTOCOL`: Triton model name
- `LABEL_MAPPING_FILE`: Path to the label mapping json file
- `NVIDIA_VISIBLE_DEVICES`: Triton model version


<br />

### Usage

The request form to get the model predictions is the following:
```
http://<host>:<port>/api/infer?url=${resource_url}&threshold=-1&iou_threshold=-1&class_id=-1$poly_points=150
```

where
-   `threshold`: Confidence score threshold
-   `iou_threshold`: NMS detection threshold
-   `class_id`: Predict only the given class id
-   `poly_points`: Upper bound of numpers of points that make the polygon
