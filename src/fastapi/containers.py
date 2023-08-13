from dependency_injector import containers, providers
from src.fastapi.services import BarcodeOCR, TorchServe


class AppContainer(containers.DeclarativeContainer):
    """
    DI Container Class
    """
    print("[AppContainer] start")
    config = providers.Configuration()
    print(f"[AppContainer] {config.barcode_ocr=}")
    barcode_ocr = providers.Singleton(
        BarcodeOCR, config.barcode_ocr,
    )
    print(f"[AppContainer] {config.torch_serve=}")
    torch_serve = providers.Singleton(
        TorchServe, config.torch_serve,
    )
    print("[AppContainer] end")