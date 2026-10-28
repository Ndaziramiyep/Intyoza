"""Local web API.   Run:  uvicorn src.api:app --port 8000 --no-access-log
Privacy by design: no database, no log of questions, nothing written to disk."""
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from . import config
from .pipeline import Intyoza

app = FastAPI(title="Intyoza", docs_url=None, redoc_url=None)   # docs off in production
bot = Intyoza()


class Question(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)


@app.post("/ask")
def ask(body: Question):
    r = bot.ask(body.question)
    # send only what the page needs; never echo the question back
    return JSONResponse({"type": r["type"], "intent": r["intent"], "title": r.get("title", ""),
                         "answer": r["answer"], "disclaimer": r["disclaimer"]},
                        headers={"Cache-Control": "no-store"})


@app.get("/health")
def health():
    return {"status": "ok", "backend": bot.matcher.backend}


@app.get("/")
def home():
    return FileResponse(config.ROOT / "app" / "index.html")
