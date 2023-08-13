import uvicorn
from fastapi import FastAPI
from omegaconf import OmegaConf
from src.fastapi.router import router
from src.fastapi import routes
from src.fastapi.containers import AppContainer


def set_routers(app: FastAPI):
    app.include_router(
        router, prefix='/router', tags=['ocr']
    )


def create_app() -> FastAPI:
    # # Инициализация DI контейнера
    container = AppContainer()
    # # Инициализация конфига
    cfg = OmegaConf.load('src/fastapi/config.yml')
    print(f"[create_app] {cfg=}")
    # # Прокидываем конфиг в наш контейнер
    container.config.from_dict(cfg)
    # # Говорим контейнеру, в каких модулях он будет внедряться
    container.wire([routes])
    app = FastAPI()
    # цепляем роутер к нашему приложению
    set_routers(app)
    return app


app = create_app()


if __name__ == '__main__':
    uvicorn.run(app, port=1234, host='0.0.0.0')
