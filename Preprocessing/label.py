import pandas as pd
import numpy as np


def load_label(label_path):
    label = pd.read_csv(label_path)
    label = label.drop(columns=['Speaker'])  # Speaker Column 제거
    label = label[label['Label'] != 0]  # Label = 0인 행 제거
    return label


def multiply_sampling_rate(label, sampling_rate):
    """
    'Start', 'End' 컬럼의 값들에 sampling_rate를 인자로 받아 곱해주는 함수 정의, 소수점 첫째 자리는 반올림하여 정수로 넘기고, 나머지 소수는 모두 버림

    :param label: 'Start', 'End' 컬럼을 갖는 DataFrame
    :param sampling_rate: 샘플링 레이트
    """
    label['Start'] = label['Start'].apply(lambda x: int(x * sampling_rate))
    label['End'] = label['End'].apply(lambda x: int(x * sampling_rate))
    return label


def create_label_column(multiplied_df, y):
    """
    multiplied_df의 'Start'~'End'에 해당하는 y의 구간(row)에 multiplied_df의 'Label'이 들어가는 column 생성,
    만약 Label이 1이나 2를 갖지 않으면 0으로 채움

    :param multiplied_df: 'Start', 'End', 'Label' 컬럼을 갖는 DataFrame
    :param y: 라벨링할 데이터
    """
    labeled_y = pd.DataFrame(np.zeros(len(y)))
    for index, row in multiplied_df.iterrows():
        start = row['Start']
        end = row['End']
        label = row['Label']
        labeled_y.loc[start:end] = label
    return labeled_y


# n_fft, hop_length를 제공받아 features.py의 함수들 처럼 데이터 길이를 축소하는 함수 정의
def adjust_length(y, n_fft=2048, hop_length=512):
    # 어떻게 줄이지
    pass


def labeling(label_path, y, sr, n_fft=2048, hop_length=512):
    # 라벨 데이터 로드
    label = load_label(label_path)

    # 샘플링 레이트에 따라 'Start', 'End' 컬럼의 값들을 곱해줌
    multiplied_label = multiply_sampling_rate(label, sr)

    # 라벨링
    labeled_y = create_label_column(multiplied_label, y)

    # 데이터 길이 조정
    # y_adjusted = adjust_length(labeled_y, n_fft, hop_length)

    return y_adjusted