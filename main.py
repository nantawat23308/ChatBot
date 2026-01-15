import uvicorn
from fastapi import FastAPI
from app.api.routes import router
from app.api.ingestion import router as ingestion_router
from prometheus_fastapi_instrumentator import Instrumentator  # Add this

app = FastAPI(title="ChatbotQA")
Instrumentator().instrument(app).expose(app)


@app.get("/")
async def root():
    return {"message": "Hello World"}


app.include_router(router)
app.include_router(ingestion_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
