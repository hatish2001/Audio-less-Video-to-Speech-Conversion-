import os
import torch
import soundfile as sf
import numpy as np
import librosa
import dlib
import cv2
import skvideo.io
import tempfile
import streamlit as st
from tqdm import tqdm
from transformers import pipeline
from datasets import load_dataset
from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass.configs import GenerationConfig
from preparation.align_mouth import landmarks_interpolate, crop_patch, write_video_ffmpeg

# Load Text-to-Speech (TTS) model and speaker embeddings
def load_model():
    """Loads the TTS model and speaker embeddings for generating speech from text."""
    synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    return synthesiser, embeddings_dataset

synthesiser, embeddings_dataset = load_model()

# Function to detect facial landmarks
def detect_landmark(image, detector, predictor):
    """Detects facial landmarks in an image using dlib's face predictor."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    coords = None
    for (_, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

# Function to preprocess video and extract region of interest (ROI)
def preprocess_video(input_video_path, output_video_path, face_predictor_path, mean_face_path):
    """Preprocesses the video by extracting mouth region of interest (ROI) for lip-reading."""
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_predictor_path)
    mean_face_landmarks = np.load(mean_face_path)
    stablePntsIDs = [33, 36, 39, 42, 45]
    videogen = skvideo.io.vread(input_video_path)
    frames = np.array([frame for frame in videogen])
    landmarks = [detect_landmark(frame, detector, predictor) for frame in tqdm(frames)]
    preprocessed_landmarks = landmarks_interpolate(landmarks)
    rois = crop_patch(input_video_path, preprocessed_landmarks, mean_face_landmarks, stablePntsIDs, (256, 256),
                      window_margin=12, start_idx=48, stop_idx=68, crop_height=96, crop_width=96)
    write_video_ffmpeg(rois, output_video_path, "/usr/bin/ffmpeg")

# Streamlit UI setup
st.title("Speech Reconstruction from Lip Movements")
st.sidebar.image("lip_reading.png")
st.sidebar.markdown("**LipReader 🎤**: Converts lip movements into speech using deep learning.")

# File uploader for video input
uploaded_file = st.file_uploader("Upload a video", type=["mp4"])
if uploaded_file and st.button('Predict'):
    # Save uploaded video
    video_path = "uploaded_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    st.video(video_path)
    
    # Define paths for required model files
    face_predictor_path = "shape_predictor_68_face_landmarks.dat"
    mean_face_path = "20words_mean_face.npy"
    mouth_roi_path = "roi.mp4"
    
    # Preprocess the video
    preprocess_video(video_path, mouth_roi_path, face_predictor_path, mean_face_path)
    st.video(mouth_roi_path)
    
    # Perform lip-sync text prediction
    os.system("python predict_lip_sync_text.py")
    with open('output_text.txt', 'r') as fp:
        hypo = fp.read()
    st.markdown(f'**Predicted Text: {hypo}**')
    
    # Convert predicted text to speech
    speech = synthesiser(hypo, forward_params={"speaker_embeddings": torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)})
    sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])
    st.audio("speech.wav", format='audio/wav')
