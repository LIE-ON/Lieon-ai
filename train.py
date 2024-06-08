# 미완성
import model.nn.esn as esn
import preprocessing.preprocessing as preprocessing
from torch.utils.data import DataLoader


# WAV 파일이 있는 디렉토리
wav_dir = '/Users/imdohyeon/Documents/PythonWorkspace/Lieon-ai/dataset/train'
wav_files = [os.path.join(wav_dir, file) for file in os.listdir(wav_dir) if file.endswith('.wav')]

"""
데이터 로드 및 전처리, 데이터로더 생성
"""
# 데이터셋 생성
dataset = preprocessing.WAVDataset(wav_iles)

# 데이터셋을 트레인 셋과 테스트 셋으로 분할
train_dataset, test_dataset = preprocessing.split_dataset(wav_dataset, train_ratio=0.8)

# 데이터로더 생성
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


"""
모델 정의, 학습 및 예측
"""
# ESN 모델 파라미터 정의


# ESN 모델 생성
model = esn.ESN(input_size, hidden_size, output_size, num_layers=num_layers,
            nonlinearity='tanh', leaking_rate=leaking_rate, spectral_radius=spectral_radius,
            w_ih_scale=1.0, lambda_reg=1.0, density=0.1, w_io=True, readout_training='gd', output_steps='all')

