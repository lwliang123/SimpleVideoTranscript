import streamlit as st
import whisper
from transformers import pipeline
import subprocess

st.title("Lecture Transcription and Summarization")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    input_video_path = "input_video.mp4"
    output_audio_path = "output_audio.wav"

    # Save the uploaded video file
    with open(input_video_path, "wb") as f:
        f.write(uploaded_file.read())

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
            # Load Whisper model and transcribe audio
            model = whisper.load_model("small")
            transcription_result = model.transcribe(output_audio_path, language='en')
            transcription = transcription_result["text"]

            st.subheader("Transcription")
            st.write(transcription)

        with st.spinner("Summarizing transcription..."):
            # Initialize summarization pipeline and summarize transcription
            summarizer = pipeline("summarization", model="t5-base")
            summary = summarizer(transcription, max_length=150, min_length=40, do_sample=False)

            st.subheader("Summary")
            st.write(summary[0]['summary_text'])

    except FileNotFoundError:
        st.error("FFmpeg is not installed or not found in the system PATH.")
        st.error("Please install FFmpeg and ensure it's accessible from the command line.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")