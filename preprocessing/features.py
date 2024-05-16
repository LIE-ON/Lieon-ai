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
import os


# Fundamental Frequency (f0) - 검증 완료, 그러나 실시간 처리를 위한 최적화 필요
def extract_f0(y, sr):
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    pitch = np.nan_to_num(f0)
    return pitch


"""
# Formants using Parselmouth
def extract_formants_praat(audio_file):
    try:
        # 오디오 파일을 Sound 객체로 변환
        Sound = parselmouth.Sound(audio_file)

        # Formant 추출
        formant = Sound.to_formant_burg(time_step=0.1)

        # Pitch 추출
        pitch = Sound.to_pitch()

        # Formant 추출에 사용된 시간을 데이터프레임으로 만들기
        df = pd.DataFrame({"times": formant.ts()})

        # F1부터 F5까지의 값을 각 시간에 대해 추출
        for idx in range(1, 6):  # F1 to F5
            df[f'F{idx}'] = df['times'].apply(lambda x: formant.get_value_at_time(formant_number=idx, time=x))

        # Pitch값 추출 (F0)
        df['F0(pitch)'] = df['times'].apply(lambda x: pitch.get_value_at_time(time=x))
        # 파일 이름 추가
        df['filename'] = os.path.basename(audio_file)

        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()  # 오류가 발생하면 빈 데이터프레임 반환
"""


# low level로 구현한 formant 추출 함수 - 검증 완료, 그러나 너무 느림 (Python 기반)
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
    flux = np.append(flux, flux[-1])  # row 수 맞추기 위해 마지막 값 복제
    return flux


# Spectral Entropy - 검증 완료
def extract_spectral_entropy(y, sr, n_fft=2048, hop_length=512):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    ps = S / np.sum(S, axis=0)
    spectral_entropy = -np.sum(ps * np.log(ps + 1e-10), axis=0)
    return spectral_entropy


# Prosody: Speech Rate and Pause Duration
def extract_prosody_features(y, sr, frame_length=0.025, hop_length=0.01):
    # Frame parameters
    n_frame_length = int(frame_length * sr)
    n_hop_length = int(hop_length * sr)
    n_frames = 1 + int((len(y) - n_frame_length) / n_hop_length)

    # Prepare to collect prosody features per frame
    speech_rates = []
    average_pauses = []

    for i in range(n_frames):
        start_sample = i * n_hop_length
        end_sample = start_sample + n_frame_length
        frame = y[start_sample:end_sample]

        # Detect silent and non-silent intervals within the frame
        intervals = librosa.effects.split(frame, top_db=20)

        # Calculate speech rate for the frame
        speech_rate = len(intervals) / (frame_length if frame_length > 0 else 1)
        speech_rates.append(speech_rate)

        # Calculate pause durations for the frame
        if len(intervals) > 1:
            pause_durations = [(intervals[j][0] - intervals[j - 1][1]) / sr for j in range(1, len(intervals))]
            average_pause = np.mean(pause_durations) if pause_durations else 0
        else:
            average_pause = 0

        average_pauses.append(average_pause)

    """
    return 후 활용 예시 : speech_rates, average_pauses = extract_prosody_features(y, sr)
    """
    return np.array(speech_rates), np.array(average_pauses)


# MFCC - 검증 완료 (n_mfcc는 추출할 feature vector의 개수, 보통 12, 13, 20 등을 사용)
# n_fft, hop_length, win_length 등의 파라미터 추가 가능 - 성능이 좋지 않으면 이 파라미터 건드려 볼 것
def extract_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=2048, hop_length=512, fmin=0, fmax=22050)
    mfcc_df = pd.DataFrame(mfcc.T, columns=[f'MFCC_{i + 1}' for i in range(mfcc.shape[0])]) # DataFrame으로 변환
    return mfcc_df