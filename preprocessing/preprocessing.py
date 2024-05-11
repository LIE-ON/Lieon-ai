import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy.signal import find_peaks
import features

# Load an audio file
audio_path = 'example.wav'  # Replace with your audio file path
y, sr = librosa.load(audio_path, sr=None)  # 우리 데이터 sr에 맞게 값 변경해야 함

# Denoising



# Feature Extraction
pitch = extract_f0(y, sr)
formants = extract_formants(audio_path)
spectral_flux = extract_spectral_flux(y, sr)
spectral_entropy = extract_spectral_entropy(y, sr)
speech_rate, pause_durations = extract_prosody(y, sr)

# Display extracted features
print(f'Pitch (f0): {pitch}')
print(f'Formants: {formants}')
print(f'Spectral Flux: {spectral_flux}')
print(f'Spectral Entropy: {spectral_entropy}')
print(f'Prosody (Speech Rate, Pause Durations): {speech_rate}, {pause_durations}')
