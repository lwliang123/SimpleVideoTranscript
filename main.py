import streamlit as st
import whisper
from transformers import pipeline
import subprocess

# Increase the maximum upload size to 500MB
st.set_option('server.maxUploadSize', 500)

st.title("Lecture Transcription and Summarization")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    with open("input_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    st.info("Extracting audio from the video...")
    # Extract audio
    subprocess.call(['ffmpeg', '-i', 'input_video.mp4', 'output_audio.wav'])

    st.info("Transcribing audio...")
    # Transcribe audio
    model = whisper.load_model("base")
    result = model.transcribe("output_audio.wav")
    transcription = result["text"]
    st.subheader("Transcription")
    st.write(transcription)

    st.info("Summarizing transcription...")
    # Summarize transcription
    summarizer = pipeline("summarization", model="t5-base")
    summary = summarizer(transcription, max_length=150, min_length=40, do_sample=False)
    st.subheader("Summary")
    st.write(summary[0]['summary_text'])
