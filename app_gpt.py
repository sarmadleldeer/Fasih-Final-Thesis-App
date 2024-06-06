import numpy as np
import streamlit as st
from st_audiorec import st_audiorec
import os
import librosa
import pickle
import tempfile
from pathlib import Path

model_path = Path(__file__).parent / "mlp_classifier.model"
logo_path = Path(__file__).parent / "fasih logo ai.png"

st.image('fasih logo ai.png', width = 400)

st.header("Speech-Emotion Recognition", divider = 'red')
st.subheader("Let's Look Inside Your Emotions :heart:")




# Read model from pickle file
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

wav_audio_data = st_audiorec()

def preprocess_recording(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return np.expand_dims(mfccs, axis=0)

def get_prediction(file_path):
    features = preprocess_recording(file_path)
    prediction = model.predict(features)
    return prediction[0]

if wav_audio_data is not None:
    # Save audio data to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_file.write(wav_audio_data)
        temp_file_path = temp_audio_file.name

    st.audio(temp_file_path, format='audio/wav')

    if st.button("Analyze your emotions now"):
        pred = get_prediction(temp_file_path)
        
        if pred == 1:
            emotion = 'Happy'
            st.balloons()
        elif pred == 2:
            emotion = 'Sad'
        elif pred == 3:
            emotion = 'Angry'
        elif pred == 4:
            emotion = 'Neutral'
        else:
            emotion = 'Unknown'
        st.balloons()
        st.divider()
        multi = f'''Your emotional state sounds like :red[{emotion}]
        but then I could be wrong too. We often mask them with a smile or two.
        '''
        st.header(multi)
        
    # Clean up temporary file
    os.remove(temp_file_path)
