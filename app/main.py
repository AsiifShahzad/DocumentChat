from dotenv import load_dotenv
load_dotenv() 
from fastapi import FastAPI
from app.api.app import router

app = FastAPI(title="DocumentChat RAG Backend")

app.include_router(router)