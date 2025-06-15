import streamlit as st
from faster_whisper import WhisperModel
import tempfile
import os

st.title("Multi-lingual Transcription using Whisper v3 (Fast & Light)")

# File uploader
audio_file = st.file_uploader("Upload your audio", type=["wav", "mp3", "m4a"])

# Load faster-whisper model
@st.cache_resource
def load_model():
    model = WhisperModel(
        model_size_or_path="large-v3",  # More accurate model
        device="cpu",                   # Change to 'cuda' if using GPU
        compute_type="int8",            # Use int8 for speed on CPU
        cpu_threads=4                   # Adjust to your system
    )
    return model

model = load_model()

# Transcription button
if st.sidebar.button("Transcribe Audio"):
    if audio_file is not None:
        st.sidebar.success("Transcribing...")

        # Save uploaded audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(audio_file.read())
            audio_file_path = os.path.abspath(temp_audio.name)

        # Run transcription
        segments, info = model.transcribe(audio_file_path, language=None)

        # Combine all segments into full transcript
        full_transcript = "\n".join(
            f"[{s.start:.2f}s - {s.end:.2f}s] {s.text}" for s in segments
        )

        st.sidebar.success("Transcription complete")
        st.markdown("### 📝 Full Transcript:")
        st.text(full_transcript)
        st.success(f"Detected Language: {info.language}")

        # Clean up
        os.remove(audio_file_path)
    else:
        st.sidebar.error("Please upload an audio file.")
