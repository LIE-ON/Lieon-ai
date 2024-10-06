# 추출된 feature들을 모두 Dataframe 형태로 반환
# 이후에 이들을 모두 한 Dataframe에 concat하고, 이 데이터를 학습시킴
# csv로 굳이 만들 필요는 없음 - 애초에 csv를 불러오는 목적이 Dataframe을 로드하기 위함이니까
# 합친 후, 바로 Model로 넘길 것

import librosa
import numpy as np
import pandas as pd
# import parselmouth
# from scipy.signal import find_peaks, lfilter, hamming
from scipy.signal import find_peaks, lfilter
from scipy.signal.windows import hamming
from scipy.io import wavfile
from scipy.fftpack import fft
from sklearn.preprocessing import MinMaxScaler
import pyworld as pw


def extract_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=2048, hop_length=512, fmin=0, fmax=22050)
    scaler = MinMaxScaler()
    mfcc_scaled = scaler.fit_transform(mfcc.T)
    mfcc_df = pd.DataFrame(mfcc_scaled, columns=[f'MFCC_{i + 1}' for i in range(mfcc.shape[0])]) # DataFrame으로 변환
    return mfcc_df


def extract_pitch(y, sr, n_fft=2048, hop_length=512):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    pitches, magnitudes = librosa.core.piptrack(S=S, sr=sr, n_fft=n_fft, hop_length=hop_length)

    # Get the maximum pitch for each frame
    pitch = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch_value = pitches[index, t]
        pitch.append(pitch_value)

    pitch = np.array(pitch)
    pitch = np.nan_to_num(pitch)
    scaler = MinMaxScaler()
    pitch_scaled = scaler.fit_transform(pitch.reshape(-1, 1))
    pitch_df = pd.DataFrame(pitch_scaled, columns=['Pitch'])
    return pitch_df


def extract_f0_pyworld(y, sr, frame_period=512 / 44100 * 1000):
    y = y.astype(np.float64)

    _f0, t = pw.dio(y, sr, frame_period=frame_period)  # Extract the initial fundamental frequency using the DIO algorithm
    f0 = pw.stonemask(y, _f0, t, sr)  # Refine the F0 estimation using the StoneMask algorithm
    scaler = MinMaxScaler()
    f0_scaled = scaler.fit_transform(f0.reshape(-1, 1))
    f0_df = pd.DataFrame(f0_scaled, columns=['F0'])
    return f0_df


def extract_spectral_flux(y, sr):
    S = np.abs(librosa.stft(y))  # STFT 계산
    flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))  # 스펙트럼 간의 차이 계산
    flux = np.append(flux, flux[-1])  # row 수 맞추기 위해 마지막 값 복제
    scaler = MinMaxScaler()
    flux_scaled = scaler.fit_transform(flux.reshape(-1, 1))
    flux_df = pd.DataFrame(flux_scaled, columns=['Spectral_Flux'])
    return flux_df


def extract_spectral_entropy(y, sr, n_fft=2048, hop_length=512):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    sum_S = np.sum(S, axis=0)
    # Check for zeros in sum_S and set them to a small value to avoid division by zero
    sum_S[sum_S == 0] = 1e-10  # or any small value to prevent division by zero

    ps = S / sum_S
    spectral_entropy = -np.sum(ps * np.log(ps + 1e-10), axis=0)
    scaler = MinMaxScaler()
    spectral_entropy_scaled = scaler.fit_transform(spectral_entropy.reshape(-1, 1))
    spectral_entropy_df = pd.DataFrame(spectral_entropy_scaled, columns=['Spectral_Entropy'])
    return spectral_entropy_df
