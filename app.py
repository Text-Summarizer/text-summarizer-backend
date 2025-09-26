from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_NAME = "facebook/bart-large-cnn"
SUMMARIZER = pipeline("summarization", model=MODEL_NAME)

MAX_LENGTH, MIN_LENGTH = (
    SUMMARIZER.model.config.max_length, 
    SUMMARIZER.model.config.min_length
)

class TextInput(BaseModel):
    text: str

@app.post("/summarize")
def summarize_text(inp: TextInput):
    summary = SUMMARIZER(inp.text, max_length=MAX_LENGTH, min_length=MIN_LENGTH, do_sample=False)
    return {"summary": summary}