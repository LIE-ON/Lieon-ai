# 미완성
import torch
import os
import Preprocessing.preprocessing as preprocessing
from Preprocessing.preprocessing import create_dataloader
from model.utils.utils import prepare_target
from torch.utils.data import DataLoader
from model.nn import ESN
import model.nn.esn as esn


# WAV 파일이 있는 디렉토리
# Parameters
wav_dir_train = '/Users/imdohyeon/Documents/PythonWorkspace/Lieon-ai/dataset/train'
wav_files_train = [os.path.join(wav_dir_train, file) for file in os.listdir(wav_dir_train) if file.endswith('.wav')]

wav_dir_test = '/Users/imdohyeon/Documents/PythonWorkspace/Lieon-ai/dataset/test'
wav_files_test = [os.path.join(wav_dir_test, file) for file in os.listdir(wav_dir_test) if file.endswith('.wav')]

"""
데이터 로드 및 전처리, 데이터로더 생성
"""
max_length = 200  # Set maximum length for padding/truncating
batch_size = 4
leaking_rate = 1.0
spectral_radius = 0.9
lambda_reg = 1e-4
washout = 10
input_size = 23  # feature 개수
hidden_size = 100
output_size = 1  # 출력 값이 0.5보다 작으면 클래스 0(거짓말 아님), 0.5보다 크면 클래스 1(거짓말)
num_layers = 1  # Reservoir 개수
w_ih_scale = 1.0  # 첫 번째 레이어의 입력 가중치 스케일
density = 1.0  # 순환 가중치 행렬의 밀도 (1이면 모든 요소가 nonzero)
readout_training = 'svd'  # Readout 학습 알고리즘 지정 (svd, cholesky, inv, gd)
output_steps = 'all'  # ridge regression 방법에서 reservoir 출력을 사용하는 방법 (last, all, mean)

# Create DataLoader
dataloader = create_dataloader(wav_files_train, max_length, batch_size, shuffle=True)


"""
모델 정의 및 학습
"""
# Initialize ESN
esn = ESN(input_size=input_size, hidden_size=hidden_size, output_size=output_size,
          num_layers=num_layers, leaking_rate=leaking_rate, spectral_radius=spectral_radius,
          lambda_reg=lambda_reg, batch_first=True)

# Training loop
num_epochs = 10
criterion = torch.nn.MSELoss()  # Example loss function
optimizer = torch.optim.Adam(esn.parameters(), lr=1e-3)


for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        # Zero gradients
        optimizer.zero_grad()

        # Prepare washout for each sample in the batch
        washout_tensor = torch.tensor([washout] * inputs.size(0))

        # Forward pass
        outputs, _ = esn(inputs, washout_tensor)

        # Remove the washout period from the outputs and targets
        outputs = outputs[:, washout:, :]
        targets = targets[:, washout:]

        min_length = min(outputs.size(1), targets.size(1))
        outputs = outputs[:, :min_length, :]  # Truncate outputs to min length
        targets = targets[:, :min_length]  # Truncate targets to min length

        outputs = outputs.squeeze(-1)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')


# Save the trained model
# torch.save(esn.state_dict(), 'esn_model_test.pth')