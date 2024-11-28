import uvicorn
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from .routers import prediction_router

app = FastAPI()
app.include_router(prediction_router)

Instrumentator().instrument(app).expose(app)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)
