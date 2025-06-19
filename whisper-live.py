import argparse
import os
from faster_whisper import WhisperModel

def load_model():
    print("🔄 Loading Whisper model...")
    model = WhisperModel(
        model_size_or_path="large-v3",  # Use "base" or "small" for faster results
        device="cpu",                   # Change to "cuda" if using GPU
        compute_type="int8",            # Or "float16" for GPU
        cpu_threads=4
    )
    print("✅ Model loaded.")
    return model

def transcribe_audio(file_path, language="ar"):
    model = load_model()
    print(f"🔍 Transcribing file: {file_path} (Language: {language})")

    segments, info = model.transcribe(file_path, language=language)

    # Prepare transcript text
    transcript_lines = [f"[{s.start:.2f}s - {s.end:.2f}s] {s.text}" for s in segments]
    full_transcript = "\n".join(transcript_lines)

    # Print to terminal
    print("📜 Transcript:\n")
    print(full_transcript)
    print(f"\n✅ Language detected: {info.language}")

    # Save to .txt file
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    txt_path = os.path.join(os.path.dirname(file_path), base_name + "_transcript.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_transcript)

    print(f"\n💾 Transcript saved to: {txt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arabic Audio Transcriber using Whisper v3")
    parser.add_argument("audio_path", help="Path to audio file (mp3, wav, m4a)")
    args = parser.parse_args()

    if os.path.exists(args.audio_path):
        transcribe_audio(args.audio_path)
    else:
        print("❌ File not found:", args.audio_path)
