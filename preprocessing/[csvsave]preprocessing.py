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
from torch.utils.data import Dataset, DataLoader


class WAVDataset(Dataset):
    def __init__(self, wav_files):
        self.wav_files = wav_files

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        wav_path = self.wav_files[idx]
        y, sr = librosa.load(wav_path, sr=44100)

        # preprocessing 과정
        mfcc = features.extract_mfcc(y, sr)
        pitch = features.extract_pitch(y, sr)
        f0_pyworld = features.extract_f0_pyworld(y, sr)
        # formants = features.extract_formants_for_frames(audio_path)
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

        return torch.tensor(concated_df, dtype=torch.float32)

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

"""
(실제 프로세스에서 사용되지 않을 코드)
실제로 코드를 수행할 때에는 csv로 저장, 이 csv를 다시 로드하여 모델에 전달하는 번거로운 과정이 포함되지 않을 것임.
-> 대신, 바로 모델에 Dataframe을 전달
"""
def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

"""
(실제 프로세스에서 사용되지 않을 코드)
실제로 코드를 수행할 때에는 csv로 저장, 이 csv를 다시 로드하여 모델에 전달하는 번거로운 과정이 포함되지 않을 것임.
-> 대신, 바로 모델에 Dataframe을 전달
"""
def process_and_save(dataloader, prefix):
    for i, batch in enumerate(dataloader):
        batch = batch.numpy()  # 텐서를 넘파이 배열로 변환
        batch_reshaped = batch.reshape(batch.shape[0], -1)  # 배치 데이터를 2D 형태로 변환
        filename = f'{prefix}_data_batch_{i}.csv'
        save_to_csv(batch_reshaped, filename)
        print(f'Saved {filename}')

"""
(실제 프로세스에서 사용되지 않을 코드)
실제로 코드를 수행할 때에는 csv로 저장, 이 csv를 다시 로드하여 모델에 전달하는 번거로운 과정이 포함되지 않을 것임.
-> 대신, 바로 모델에 Dataframe을 전달
"""
class CSVDataset(Dataset):
    def __init__(self, csv_files):
        self.dataframes = [pd.read_csv(file) for file in csv_files]
        self.merged_df = self.merge_feature_dataframes(self.dataframes)

    def merge_feature_dataframes(self, dfs):
        return pd.concat(dfs, axis=0)

    def __len__(self):
        return len(self.merged_df)

    def __getitem__(self, idx):
        row = self.merged_df.iloc[idx].values
        return torch.tensor(row, dtype=torch.float32)



# WAV 파일이 있는 디렉토리
wav_dir = 'dataset/train'
wav_files = [os.path.join(wav_dir, file) for file in os.listdir(wav_dir) if file.endswith('.wav')]

# 데이터셋 생성
dataset = WAVDataset(wav_files)

# 데이터셋을 트레인 셋과 테스트 셋으로 분할
train_dataset, test_dataset = split_dataset(wav_dataset, train_ratio=0.8)

# 데이터로더 생성
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


# train&test 데이터셋을 처리하고 CSV로 저장
process_and_save(train_dataloader, 'train')
process_and_save(test_dataloader, 'test')


# CSV 파일이 있는 디렉토리 설정
train_csv_dir = 'path_to_train_csv_files'
test_csv_dir = 'path_to_test_csv_files'
train_csv_files = [os.path.join(train_csv_dir, file) for file in os.listdir(train_csv_dir) if file.endswith('.csv')]
test_csv_files = [os.path.join(test_csv_dir, file) for file in os.listdir(test_csv_dir) if file.endswith('.csv')]

# 데이터셋 생성
train_csv_dataset = CSVDataset(train_csv_files)
test_csv_dataset = CSVDataset(test_csv_files)

# 데이터로더 생성
train_csv_dataloader = DataLoader(train_csv_dataset, batch_size=32, shuffle=True, num_workers=4)
test_csv_dataloader = DataLoader(test_csv_dataset, batch_size=32, shuffle=False, num_workers=4)


"""
# ESN 모델 파라미터 정의
from torchesn.nn import ESN
from torchesn.utils import prepare_target

input_size = train_csv_dataset[0].shape[0]  # 피처 수
hidden_size = 100  # 예시 은닉층 크기
output_size = 1  # 예시 출력 크기
washout = 0  # 예시 워시오트 기간


# ESN 모델 생성
model = ESN(input_size, hidden_size, output_size)

# 학습을 위한 타겟 데이터 준비 (타겟 데이터가 있는 경우)
# 여기서는 단순히 예시로 랜덤 타겟 데이터를 생성
target_data = torch.randn(len(train_csv_dataset), output_size)
flat_target = prepare_target(target_data.unsqueeze(1), [len(train_csv_dataset)], [washout])

# 모델 학습
for batch in train_csv_dataloader:
    batch = batch.unsqueeze(1)  # 배치 차원 추가
    output, _ = model(batch, [washout], None, flat_target)
    model.fit()

# 예측
for batch in test_csv_dataloader:
    batch = batch.unsqueeze(1)
    output, _ = model(batch, [washout], None)
    print(output)
    break
"""