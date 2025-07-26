# backend/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from yt_utils import download_audio_from_youtube
from whisper_utils import transcribe_audio
from llm_utils import run_qa
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# Allow frontend to access backend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class YouTubeRequest(BaseModel):
    url: str
    question: str

@app.post("/youtube-qa/")
def youtube_qa(req: YouTubeRequest):
    try:
        audio_file = download_audio_from_youtube(req.url)
        transcript = transcribe_audio(audio_file)

        with open("transcribed.txt", "w", encoding="utf-8") as f:
            f.write(transcript)

        answer = run_qa("transcribed.txt", req.question)
        return {"answer": answer, "transcript": transcript}
    except Exception as e:
        return {"error": str(e)}

