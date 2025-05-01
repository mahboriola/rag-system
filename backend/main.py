from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_utils.timing import add_timing_middleware

from api import consult, ingestion
from config.startup import lifespan
from services.logs import logger

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

add_timing_middleware(app, record=logger.info, prefix="app")

app.include_router(consult.router)
app.include_router(ingestion.router)
