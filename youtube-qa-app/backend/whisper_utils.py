import whisper
import torch

def transcribe_audio(file_path: str) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("turbo")
    model = model.to(device)
    result = model.transcribe(file_path)
    return result["text"]
