import librosa
import numpy as np
import pandas as pd
import Preprocessing.features as features  # features.py 파일을 import
from Preprocessing.label import labeling
import torch
from torch.utils.data import Dataset, DataLoader

import os
import boto3

s3 = boto3.client('s3')


def download_s3_file(s3_path, local_dir="/tmp"):
    """
    S3 경로에서 파일을 로컬로 다운로드
    :param s3_path: S3 파일 경로 (s3://bucket/key) 또는 로컬 파일 경로
    :param local_dir: 로컬에 저장할 디렉토리
    :return: 로컬 파일 경로
    """
    if s3_path.startswith("s3://"):
        # S3 경로에서 버킷명과 키 추출
        s3_path = s3_path.replace("s3://", "")
        bucket, key = s3_path.split('/', 1)
        # print('bucket:', bucket, ', key:', key) debugging code

        # 로컬 파일 경로 설정
        local_file_path = os.path.join(local_dir, os.path.basename(key))

        # S3에서 파일 다운로드
        s3.download_file(bucket, key, local_file_path)
        return local_file_path
    else:
        # 로컬 경로일 경우
        return s3_path


class WAVDataset(Dataset):
    def __init__(self, wav_path, label_path, max_length):
        """
        초기화 메서드
        :param wav_path: 처리할 WAV 파일의 경로 리스트
        :param max_length: 각 샘플의 최대 길이
        """
        self.wav_path = wav_path
        self.label_path = label_path
        self.max_length = max_length

    def __len__(self):
        """
        데이터셋의 길이 반환
        :return: 데이터셋에 있는 WAV 파일의 수
        """
        return len(self.wav_path)

    def __getitem__(self, idx):
        """
        인덱스에 해당하는 데이터 반환
        :param idx: 인덱스
        ====================================================
        - Preprocessing (features.py 이용)
        - 피처 병합
        - 피처 데이터(X)와 라벨(y) 분리
        ====================================================
        :return: 피처 데이터(X)와 라벨(y)을 텐서 형태로 반환
        """
        wav_path = self.wav_path[idx]
        wav_s3_path = download_s3_file(wav_path)
        y, sr = librosa.load(wav_s3_path, sr=44100)
        label_path = self.label_path[idx]
        label_s3_path = download_s3_file(label_path)

        # debugging code
        print(f"Processing index: {idx + 1} / {len(self.wav_path)}")

        # 단일 데이터에 대해 피처 추출 및 pad_or_truncate 적용
        mfcc = self.pad_or_truncate(features.extract_mfcc(y, sr).values)
        pitch = self.pad_or_truncate(features.extract_pitch(y, sr).values)
        f0_pyworld = self.pad_or_truncate(features.extract_f0_pyworld(y, sr).values)
        spectral_flux = self.pad_or_truncate(features.extract_spectral_flux(y, sr).values)
        spectral_entropy = self.pad_or_truncate(features.extract_spectral_entropy(y, sr).values)
        labeled = self.pad_or_truncate(labeling(label_s3_path, y, sr).values)

        features_dict = {
            'mfcc': pd.DataFrame(mfcc),
            'pitch': pd.DataFrame(pitch),
            'f0_pyworld': pd.DataFrame(f0_pyworld),
            'spectral_flux': pd.DataFrame(spectral_flux),
            'spectral_entropy': pd.DataFrame(spectral_entropy),
            'label': pd.DataFrame(labeled)
        }

        concatenated_df = self.merge_features(features_dict)

        # 라벨과 나머지 데이터 분리
        X = concatenated_df.iloc[:, :-1]
        y = concatenated_df.iloc[:, -1]

        # X와 y를 텐서로 변환
        X = torch.tensor(X.values, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.long)

        return X, y

    # 피처 병합 함수 (merge_features)를 통해 병합
    def merge_features(self, features_dict):
        df_list = []
        for key, df in features_dict.items():
            df_list.append(df)
        concatenated_df = pd.concat(df_list, axis=1)
        concatenated_df = concatenated_df.fillna(0)  # NaN 값은 0으로 채웁니다.
        return concatenated_df

    def pad_or_truncate(self, features):
        """
        피처별로 패딩 또는 잘라내기를 적용 (다차원 데이터 처리)
        :param features: 피처 데이터 (1차원 또는 2차원 배열)
        :return: 패딩 또는 잘라내기 후의 데이터
        """
        # 피처 데이터가 Series일 경우 넘파이 배열로 변환
        if isinstance(features, pd.Series):
            features = features.values

        # 피처 데이터가 2차원 배열일 경우, 각 차원의 패딩을 맞추기 위한 코드
        if features.ndim == 2:
            length, feature_dim = features.shape

            # max_length보다 크면 자르기
            if length > self.max_length:
                return features[:self.max_length, :]

            # max_length보다 짧으면 패딩 추가
            elif length < self.max_length:
                pad_width = self.max_length - length
                padding = np.zeros((pad_width, feature_dim))
                return np.concatenate([features, padding], axis=0)

        # 1차원 배열일 경우 기존 방식 사용
        else:
            length = len(features)

            if length > self.max_length:
                return features[:self.max_length]
            elif length < self.max_length:
                pad_width = self.max_length - length
                padding = np.zeros(pad_width)
                return np.concatenate([features, padding])

        return features


def create_dataloader(wav_path, label_path, max_length, batch_size, shuffle=True):
    dataset = WAVDataset(wav_path, label_path, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=shuffle)
    return dataloader
