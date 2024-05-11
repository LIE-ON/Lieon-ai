import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy.signal import find_peaks


# Fundamental Frequency (f0) and Pitch
def extract_f0(y, sr):
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    pitch = np.nan_to_num(f0)
    return pitch


# Formants using Parselmouth
def extract_formants(audio_path):  # librosa 사용하지 않으므로 바로 audio_path 로드
    sound = parselmouth.Sound(audio_path)
    formant = call(sound, "To Formant (burg)", 0.025, 5, 5500, 0.02, 50)
    formants = {}
    for i in range(1, 6):
        formants[f"formant_{i}"] = np.array([call(formant, "Get value at time", i, t, 'Hz') for t in sound.ts()])
    return formants


# Spectral Flux
def extract_spectral_flux(y, sr):
    spectral_flux = np.diff(librosa.feature.rms(y=y).flatten()) ** 2
    return spectral_flux


# Spectral Entropy
# n_fft와 hop_length는 데이터에 맞게 커스터마이징할 것
def extract_spectral_entropy(y, sr, n_fft=2048, hop_length=512):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    ps = S / np.sum(S, axis=0)
    spectral_entropy = -np.sum(ps * np.log(ps + 1e-10), axis=0)
    return spectral_entropy


# Prosody: Speech Rate and Pause Duration
def extract_prosody(y, sr):
    intervals = librosa.effects.split(y, top_db=20)
    pause_durations = [(intervals[i][0] - intervals[i - 1][1]) / sr for i in range(1, len(intervals))]
    speech_rate = len(intervals) / (sum([end - start for start, end in intervals]) / sr)
    return speech_rate, pause_durations

# MFCC (n_mfcc는 추출할 feature vector의 개수, 보통 12, 13, 20 등을 사용)
# n_fft, hop_length, win_length 등의 파라미터 추가 가능 - 성능이 좋지 않으면 이 파라미터 건드려 볼 것
def extract_mfcc(y, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc