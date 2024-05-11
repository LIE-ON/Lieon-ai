import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy.signal import find_peaks
import features

# Load an audio file
audio_path = 'example.wav'  # Replace with your audio file path
y, sr = librosa.load(audio_path, sr=None)  # 우리 데이터 sr에 맞게 값 변경해야 함

