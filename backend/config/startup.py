from contextlib import asynccontextmanager
from fastapi import FastAPI

from services.logs import logger
from services.vector_database import VectorDatabase


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Checking Vector DB Collection")
        await VectorDatabase.assert_collection()
        logger.info("Vector DB Collection is ready")
    except Exception as e:
        logger.error(f"Error during Vector DB Collection check: {e}")
        await VectorDatabase.create_collection()
        logger.info("Vector DB Collection created")
    
    yield