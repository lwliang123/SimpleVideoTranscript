import streamlit as st
import whisper
from transformers import pipeline
import subprocess
import tempfile
import os
import time  # Import the time module

st.title("Lecture Transcription and Summarization")

# Define maximum file size (e.g., 100MB)
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

# File uploader
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("small")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="t5-base")

if uploaded_file is not None:
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error("File size exceeds the 100MB limit. Please upload a smaller file.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(uploaded_file.read())
            input_video_path = temp_video.name

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            output_audio_path = temp_audio.name

        try:
            with st.spinner("Extracting audio from the video..."):
                # Correct FFmpeg command to extract audio
                ffmpeg_command = [
                    'ffmpeg',
                    '-i', input_video_path,          # Input file
                    '-vn',                           # Disable video
                    '-acodec', 'pcm_s16le',          # Audio codec
                    '-ar', '44100',                  # Audio sampling rate
                    '-ac', '2',                      # Number of audio channels
                    output_audio_path                # Output audio file
                ]

                # Execute FFmpeg command with error handling
                result = subprocess.run(
                    ffmpeg_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                if result.returncode != 0:
                    st.error("Error extracting audio with FFmpeg:")
                    st.error(result.stderr)
                    st.stop()  # Stop further execution
                else:
                    st.success("Audio extracted successfully!")

            with st.spinner("Transcribing audio..."):
                # Start the timer before transcription
                start_time = time.perf_counter()

                # Load Whisper model and transcribe audio with English language specification
                model = load_whisper_model()
                transcription_result = model.transcribe(output_audio_path, language='en')
                transcription = transcription_result["text"]

                # End the timer after transcription
                end_time = time.perf_counter()

                # Calculate elapsed time
                elapsed_time = end_time - start_time

                # Display the transcription
                st.subheader("Transcription")
                st.write(transcription)

                # Display the elapsed time
                minutes, seconds = divmod(elapsed_time, 60)
                st.write(f"**Transcription completed in {int(minutes)} minutes and {seconds:.2f} seconds.**")

            with st.spinner("Summarizing transcription..."):
                # Initialize summarization pipeline and summarize transcription
                summarizer = load_summarizer()
                summary = summarizer(transcription, max_length=150, min_length=40, do_sample=False)

                st.subheader("Summary")
                st.write(summary[0]['summary_text'])

        except FileNotFoundError:
            st.error("FFmpeg is not installed or not found in the system PATH.")
            st.error("Please install FFmpeg and ensure it's accessible from the command line.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
        finally:
            # Clean up temporary files
            if os.path.exists(input_video_path):
                os.remove(input_video_path)
            if os.path.exists(output_audio_path):
                os.remove(output_audio_path)
