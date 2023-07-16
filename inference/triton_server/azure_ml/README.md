# Deployment on Azure Machine Learning

## Pre-requisites

```
cd inference/triton_server
```

Set the environment for AML:
```
export RESOURCE_GROUP=Dhruva-prod
export WORKSPACE_NAME=dhruva--central-india
export DOCKER_REGISTRY=dhruvaprod
```

Also remember to edit the `yml` files accordingly.

## Registering the model

```
az ml model create --file azure_ml/model.yml --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME
```

## Pushing the docker image to Container Registry

```
az acr login --name $DOCKER_REGISTRY
docker tag indictrans2_triton $DOCKER_REGISTRY.azurecr.io/nmt/triton-indictrans-v2:latest
docker push $DOCKER_REGISTRY.azurecr.io/nmt/triton-indictrans-v2:latest
```

## Creating the execution environment

```
az ml environment create -f azure_ml/environment.yml -g $RESOURCE_GROUP -w $WORKSPACE_NAME
```

## Publishing the endpoint for online inference

```
az ml online-endpoint create -f azure_ml/endpoint.yml -g $RESOURCE_GROUP -w $WORKSPACE_NAME
```

Now from the Azure Portal, open the Container Registry, and grant ACR_PULL permission for the above endpoint, so that it is allowed to download the docker image.

## Attaching a deployment

```
az ml online-deployment create -f azure_ml/deployment.yml --all-traffic -g $RESOURCE_GROUP -w $WORKSPACE_NAME
```

## Testing if inference works

1. From Azure ML Studio, go to the "Consume" tab, and get the endpoint domain (without `https://` or trailing `/`) and an authentication key.
2. In `client.py`, enable `ENABLE_SSL = True`, and then set the `ENDPOINT_URL` variable as well as `Authorization` value inside `HTTP_HEADERS`.
3. Run `python3 client.py`
