from fastapi import FastAPI

from routers import images

app = FastAPI()
app.include_router(images.router)
