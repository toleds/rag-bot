import middleware
from entrypoints.router.v1 import router

from fastapi import FastAPI

# Create the FastAPI app
app = FastAPI(title="RAG-Bot", version="1.0.0", base_path="/v1")

# Include the router into the FastAPI app
app.include_router(router.router)

# Configure middleware
middleware.configure(app)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
