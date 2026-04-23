from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from app.routers.reasoning import router

app = FastAPI()

app.include_router(router)