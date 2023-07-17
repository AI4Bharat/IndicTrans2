# Triton server

## Building the image

```
cd indicTrans2/inference/
docker build -f triton_server/Dockerfile -t indictrans2_triton .
```

## Running the container

Place the `en-indic` and `indic-en` checkpoint folders into `indicTrans2/checkpoints` directory

Then start the server by:
```
docker run --shm-size=256m --gpus=1 --rm -v ${PWD}/../checkpoints/:/models/checkpoints -p 8000:8000 -t indictrans2_triton
```

## Sample client

- Do `pip install tritonclient[all] gevent` first.
- Then `python3 triton_server/client.py`
