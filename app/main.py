from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

from .routers import images, video

app = FastAPI()
app.include_router(images.router)
app.include_router(video.router)


origins = [
    '*'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
