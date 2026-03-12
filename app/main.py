from dotenv import load_dotenv
load_dotenv() 
from fastapi import FastAPI
from app.api.app import router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="DocumentChat RAG Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # replace "*" with your Vercel URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)