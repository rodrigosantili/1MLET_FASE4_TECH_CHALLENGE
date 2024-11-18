import uvicorn
from fastapi import FastAPI

from routers import prediction_router


print("Starting the application")
app = FastAPI()
app.include_router(prediction_router)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)
