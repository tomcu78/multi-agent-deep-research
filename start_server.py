"""Helper entrypoint to launch the Deep Research FastAPI server on port 8001."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8001, reload=True)
