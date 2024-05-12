# 추출된 feature들을 모두 Dataframe 형태로 반환
# 이후에 이들을 모두 한 Dataframe에 concat하고, 이 데이터를 학습시킴
# csv로 굳이 만들 필요는 없음 - 애초에 csv를 불러오는 목적이 Dataframe을 로드하기 위함이니까
# 합친 후, 바로 Model로 넘길 것


import librosa
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
from scipy.signal import find_peaks, lfilter, hamming
from scipy.io import wavfile
from scipy.fftpack import fft


# Fundamental Frequency (f0) - 검증 완료, 그러나 실시간 처리를 위한 최적화 필요
def extract_f0(y, sr):
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    pitch = np.nan_to_num(f0)
    return pitch


"""
The code below is no longer used.
------------------------------------------------------------
# Formants using Parselmouth

def extract_formants(audio_file):
    ""
    Extracts the first three formants (F1, F2, F3) from the given audio file.

    Args:
    audio_file (str): Path to the audio file.

    Returns:
    list of tuples: Each tuple contains the time, F1, F2, and F3 values.
    ""
    # Load the audio file
    sound = parselmouth.Sound(audio_file)

    # Analyze formants
    formant = call(sound, "To Formant (burg)", 0.0025, 5, 5500, 0.025, 50)

    # Gather formant data
    num_points = call(formant, "Get number of frames")
    formant_data = []
    for i in range(1, num_points + 1):
        t = call(formant, "Get time from frame number", i)
        f1 = call(formant, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formant, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formant, "Get value at time", 3, t, 'Hertz', 'Linear')
        formant_data.append((t, f1, f2, f3))

    return formant_data
"""

# low level로 구현한 formant 추출 함수 - 검증 완료, 그러나 너무 느림
def extract_formants_for_frames(audio_file, frame_length=0.025, hop_length=0.01):
    sample_rate, signal = wavfile.read(audio_file)
    if len(signal.shape) == 2:
        signal = signal[:, 0]

    # Frame parameters
    n_frame = int(frame_length * sample_rate)
    n_hop = int(hop_length * sample_rate)
    n_frames = 1 + int((len(signal) - n_frame) / n_hop)

    # Buffer to hold formant data
    formants_buffer = []

    for i in range(n_frames):
        start_sample = i * n_hop
        frame = signal[start_sample:start_sample + n_frame]
        windowed = frame * hamming(n_frame)
        lpc_order = 2 + sample_rate // 1000
        A = librosa.lpc(windowed, order=lpc_order)
        rts = np.roots(A)
        rts = rts[np.imag(rts) >= 0]
        angz = np.arctan2(np.imag(rts), np.real(rts))
        formants = angz * (sample_rate / (2 * np.pi))
        formants_sorted = np.sort(formants)

        # Handling variable number of formants
        formants_to_use = formants_sorted[:5]  # Take the first 5 formants
        if len(formants_to_use) < 5:
            formants_to_use = np.pad(formants_to_use, (0, 5 - len(formants_to_use)), 'constant',
                                     constant_values=(np.nan,))

        formants_buffer.append(formants_to_use)

    return np.array(formants_buffer)


# Spectral Flux - 검증 완료
def extract_spectral_flux(y, sr):
    S = np.abs(librosa.stft(y))  # STFT 계산
    flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))  # 스펙트럼 간의 차이 계산
    return flux


# Spectral Entropy - 검증 완료
def extract_spectral_entropy(y, sr, n_fft=2048, hop_length=1024):
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

# MFCC - 검증 완료 (n_mfcc는 추출할 feature vector의 개수, 보통 12, 13, 20 등을 사용)
# n_fft, hop_length, win_length 등의 파라미터 추가 가능 - 성능이 좋지 않으면 이 파라미터 건드려 볼 것
def extract_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=1024, fmin=0, fmax=8000)
    mfcc_df = pd.DataFrame(mfcc.T, columns=[f'MFCC_{i + 1}' for i in range(mfcc.shape[0])]) # DataFrame으로 변환
    return mfcc_df