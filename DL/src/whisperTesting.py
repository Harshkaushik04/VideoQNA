import whisper
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("turbo")  
model = model.to(device)
print(f"device:{device}")
result = model.transcribe("../inputFiles/＂Good People Are Weak＂ ｜ Nietzsche's Most Dangerous Idea [iqwVDFY6J6c].mp3")
print(result["text"])
print("len:",len(result["text"]))
