from fastapi import FastAPI

from .routers import images, video

app = FastAPI()
app.include_router(images.router)
app.include_router(video.router)
