import requests
import typing as tp
from fastapi import Depends, File
from dependency_injector.wiring import Provide, inject

from src.fastapi.router import router
from src.fastapi.services import BarcodeOCR, TorchServe
from src.fastapi.containers import AppContainer


@router.get('/health_check')
def health_check():
    return {'status': 'OK'}


@router.post('/tserve_status')
@inject
def tserve_status(
    service: TorchServe = Depends(Provide[AppContainer.torch_serve]),
):
    print("[tserve_status] start")
    result = service.health_check()
    print("[tserve_status] end")
    return result


@router.post('/barcode_ocr')
@inject
def predict(
    image: bytes = File(),
    service: BarcodeOCR = Depends(Provide[AppContainer.barcode_ocr]),
) -> tp.List[str]:
    result = service.predict(image)
    return result
    # response = requests.get(
    #     "http://127.0.0.1:8080/wfpredict/wf", 
    #     data=image
    # )
    # return response
    
