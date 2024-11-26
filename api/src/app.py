from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from routers import prediction_router


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     Instrumentator().instrument(app).expose(app)
#
#     yield

#app = FastAPI(lifespan=lifespan)

app = FastAPI()
app.include_router(prediction_router)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)
