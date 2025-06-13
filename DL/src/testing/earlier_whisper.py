import whisper
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("turbo")  
model = model.to(device)
print(f"device:{device}")
result = model.transcribe("../../inputFiles/[CS61C FA20] Lecture 12.3 - RISC-V Instruction Formats II： J-Format [hkVUmw460Kw].mp3")
print(result["text"])
print("len:",len(result["text"]))