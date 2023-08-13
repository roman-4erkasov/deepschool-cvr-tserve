# deepschool-cvr-tserve

Pipeline for Barcode number recognition. Consisits of two models:
- Barcode detection model based on YOLOv5
- OCR model based on CRNN

Pipeline consists of two services:
1. FastAPI service to interact with users
2. TorchServe to orchestracte the models. Models are combined into TorchServe  Workflow.

Pipeline accepts photos of barcodes.
And the result of the pipeline is json of the following format:
```json
{
  "barcodes": [
    {
      "bbox": {
        "x_min": 211,
        "x_max": 620,
        "y_min": 477,
        "y_max": 776
      },
      "value": "1219181"
    },
    {
      "bbox": {
        "x_min": 311,
        "x_max": 520,
        "y_min": 877,
        "y_max": 976
      },
      "value": "1219181"
    } 
  ]
}
```

## TODO
 - [ ] Add FastAPI to Docker Compose
 - [ ] Add DVC
 - [ ] Add Monitoring
 - [ ] Move model dependecies from Dockerfile to requirements for MAR
 - [ ] Add PyTest
 - [ ] Try Ansible for automatic deployment

## Getting started


1. Download models in torchscript format (call me).
2. Build TorchServe images `make build`
3. Start up docker compose: `make run_tserve`
4. Deploy models to TorchServe: `make deploy`
5. Start up FastAPI service: `make run_fastapi`
6. Go to `http://127.0.0.1:1234/docs` 



