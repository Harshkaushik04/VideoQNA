# import whisper
# import torch

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = whisper.load_model("turbo")  
# model = model.to(device)
# print(f"device:{device}")
# result = model.transcribe("../inputFiles/Sex and Christianity： A Tumultuous History - Diarmaid MacCulloch [78HSOUODK3E].mp3")
# print(result["text"])
# print("len:",len(result["text"]))



# import whisperx
# # Set model + device
# device = "cuda"
# model = whisperx.load_model("large-v3-turbo", device)

# # Transcribe with word timestamps
# audio_path = "../inputFiles/[CS61C FA20] Lecture 12.3 - RISC-V Instruction Formats II： J-Format [hkVUmw460Kw].mp3"
# transcription = model.transcribe(audio_path)

# print("Segments:")
# for segment in transcription["segments"]:
#     print(f"[{segment['start']:.2f} - {segment['end']:.2f}]: {segment['text']}")

# print("keys:")
# print(transcription.keys())
# print("\nFirst few words with timestamps:")
# for word in transcription["segments"][:5]:
#     print(f"{word['text']} ({word['start']:.2f}s - {word['end']:.2f}s)")

# # Optional: Run diarization
# diarize_model = whisperx.diarize.DiarizationPipeline("pyannote/speaker-diarization-3.1",use_auth_token="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",device=device)
# print("hi")
# diarization_segments = diarize_model(audio_path)
# print("hi again")
# # Assign speaker labels
# transcription = whisperx.assign_word_speakers(transcription["segments"], diarization_segments)

# print("\nSpeaker-labeled segments:")
# for segment in transcription:
#     num_starting_hours=(int)(segment['start']/3600)
#     num_starting_minutes=(int)((segment['start']-num_starting_hours*3600)/60)
#     num_starting_seconds=(int)(segment['start']-3600*num_starting_hours-60*num_starting_minutes)
#     num_ending_hours=(int)(segment['end']/3600)
#     num_ending_minutes=(int)((segment['end']-num_ending_hours*3600)/60)
#     num_ending_seconds=(int)(segment['end']-3600*num_ending_hours-60*num_ending_minutes)
#     print(f"[{num_starting_hours}:{num_starting_minutes}:{num_starting_seconds}-{num_ending_hours}:{num_ending_minutes}:{num_ending_seconds}] Speaker {segment['speaker']}: {segment['text']}")
import whisperx

device = "cuda"

# 1. Load model and transcribe
model = whisperx.load_model("large-v3-turbo", device)
audio_path = "../inputFiles/CRAZY UNO REVERSE Moment…💀｜ Vijay Mallya Son ANGRY, Harsh ROAST Pakistan, Ashish REQUEST, Mythpat ｜ [NQNZeIYSRn8].mp3"
audio = whisperx.load_audio(audio_path)
result = model.transcribe(audio,language="en",task="translate")

# 2. Align output
model_a, metadata = whisperx.load_align_model(
    language_code=result["language"], 
    device=device
)
result = whisperx.align(
    result["segments"], 
    model_a, 
    metadata, 
    audio, 
    device, 
    return_char_alignments=False
)

# 3. Diarization with word-level assignment
diarize_model = whisperx.diarize.DiarizationPipeline(  # Changed to diarize submodule
    use_auth_token="HF_TOKEN",
    device=device
)
diarize_segments = diarize_model(audio)
result = whisperx.assign_word_speakers(diarize_segments,result)  # Using word-level assignment
def print_merged_transcript(segments, interval=30):
    current_window_end = interval
    current_speaker = None
    current_text = []
    speaker_map = {}
    speaker_count = 1

    for seg in sorted(segments, key=lambda x: x['start']):
        speaker = seg.get('speaker', 'UNKNOWN')
        if speaker not in speaker_map:
            speaker_map[speaker] = f"S{speaker_count}"
            speaker_count += 1
        short_speaker = speaker_map[speaker]

        # Merge segments within window
        if seg['start'] < current_window_end:
            if short_speaker == current_speaker:
                current_text.append(seg['text'].strip('., '))
            else:
                if current_speaker:
                    print(f"[{int(current_window_end-interval)}s] {current_speaker}: {' '.join(current_text)}")
                current_speaker = short_speaker
                current_text = [seg['text'].strip('., ')]
        else:
            if current_text:
                print(f"[{int(current_window_end-interval)}s] {current_speaker}: {' '.join(current_text)}")
            current_window_end += interval
            current_speaker = short_speaker
            current_text = [seg['text'].strip('., ')]

    # Print final segment
    if current_text:
        print(f"[{int(current_window_end-interval)}s] {current_speaker}: {' '.join(current_text)}")


print("\nFormatted Transcript:")
print_merged_transcript(result["segments"])
