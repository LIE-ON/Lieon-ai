import librosa
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
from scipy.signal import find_peaks, lfilter, hamming
from scipy.io import wavfile
from scipy.fftpack import fft
import features  # features.py 파일을 import
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class WAVDataset(Dataset):
    def __init__(self, wav_files):
        """
        초기화 메서드
        :param wav_files: 처리할 WAV 파일의 경로 리스트
        """
        self.wav_files = wav_files

    def __len__(self):
        """
        데이터셋의 길이 반환
        :return: 데이터셋에 있는 WAV 파일의 수
        """
        return len(self.wav_files)

    def __getitem__(self, idx):
        """
        인덱스에 해당하는 데이터 반환
        :param idx: 인덱스
        ====================================================
        - Preprocessing (features.py 이용)
        - 피처 병합 (merge_features 메서드 이용)
        - 피처 데이터(X)와 라벨(y) 분리
        ====================================================
        :return: 피처 데이터(X)와 라벨(y)을 텐서 형태로 반환
        """
        wav_path = self.wav_files[idx]
        y, sr = librosa.load(wav_path, sr=44100)

        # preprocessing 과정
        mfcc = features.extract_mfcc(y, sr)
        pitch = features.extract_pitch(y, sr)
        f0_pyworld = features.extract_f0_pyworld(y, sr)
        spectral_flux = features.extract_spectral_flux(y, sr)
        spectral_entropy = features.extract_spectral_entropy(y, sr)

        # 추출된 feature 병합한 dataframe을 concated_df로 선언 후, return
        # 피처 병합
        features_dict = {
            'mfcc': mfcc,
            'pitch': pitch,
            'f0_pyworld': f0_pyworld,
            'spectral_flux': spectral_flux,
            'spectral_entropy': spectral_entropy
        }
        concated_df = self.merge_features(features_dict)

        # 라벨과 나머지 데이터 분리
        X = concated_df.iloc[:, :-1].values  # 마지막 열을 제외한 나머지
        y = concated_df.iloc[:, -1].values  # 마지막 열

        # X와 y를 텐서로 변환
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return X, y

    def merge_features(self, features_dict):
        # 피처들을 하나의 데이터프레임으로 병합
        df_list = []
        for key, df in features_dict.items():
            # 각 DataFrame의 행 수를 통일
            df_list.append(df)

        # 열 방향으로 병합
        concated_df = pd.concat(df_list, axis=1)
        return concated_df


def split_dataset(dataset, train_ratio=0.8):
    train_size = int(train_ratio * len(dataset))  # 트레인 셋 비율
    test_size = len(dataset) - train_size  # 나머지를 테스트 셋으로 사용
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset



