import os
from fastapi import FastAPI
from dotenv import load_dotenv
import uvicorn

load_dotenv()

app = FastAPI(
    title=os.getenv("API_TITLE", "Bluetti Monitor API"),
    description="Monitor Bluetti solar generator battery status via ESP32 webcam",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Bluetti Monitor is running"}

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug
    )