"""FastAPI application entry point."""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging

from app.core.config import cors_options, settings
from app.api.routes import router
from app.observability.tracing import start_tracing, stop_tracing

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    logger.info("Starting MARA Research Platform...")
    start_tracing()
    try:
        yield
    finally:
        stop_tracing()
        logger.info("Shutting down MARA...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Multi-Agent Research Assistant with LangGraph",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    **cors_options(settings.BACKEND_CORS_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


app.include_router(router, prefix=settings.API_V1_STR)


@app.get("/")
async def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": f"{settings.API_V1_STR}/health",
        "deployment_mode": "stateless-memory"
    }

# Render ignores this.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "development"
    )
