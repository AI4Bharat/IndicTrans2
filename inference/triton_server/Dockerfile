ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:22.12-py3
FROM ${BASE_IMAGE}

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONIOENCODING=utf8

WORKDIR /home

WORKDIR /home/indicTrans2
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY download.py .
RUN python3 download.py

COPY . ./inference

WORKDIR /home/
COPY ./triton_server/triton_repo ./triton_repo

CMD ["tritonserver", "--model-repository=/home/triton_repo", "--log-verbose=2", "--strict-model-config=false", "--http-port=8000", "--grpc-port=8001", "--metrics-port=8002"]
EXPOSE 8000
EXPOSE 8001
EXPOSE 8002
