# train3.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from sklearn.metrics import accuracy_score, f1_score

from Preprocessing.preprocessing import create_dataloader
from Classifier.nn.esn import ESN


def get_file_paths(wav_dir, label_dir):
    wav_files = [file for file in os.listdir(wav_dir) if file.endswith('.wav')]
    wav_files.sort()  # 파일 이름을 정렬하여 매칭을 보장
    label_files = [file for file in os.listdir(label_dir) if file.endswith('.csv')]
    label_files.sort()

    wav_paths = [os.path.join(wav_dir, file) for file in wav_files]
    label_paths = [os.path.join(label_dir, file) for file in label_files]
    return wav_paths, label_paths


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    all_preds = []
    all_labels = []
    total_loss = 0
    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)

        batch_size_actual = X.size(0)
        washout = torch.zeros(batch_size_actual, dtype=torch.int64).to(device)

        outputs, _ = model(X, washout=washout)

        outputs = outputs.reshape(-1, model.output_size)
        y = y.reshape(-1)

        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, accuracy, f1


def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device)

            batch_size_actual = X.size(0)
            washout = torch.zeros(batch_size_actual, dtype=torch.int64).to(device)

            outputs, _ = model(X, washout=washout)

            outputs = outputs.reshape(-1, model.output_size)
            y = y.reshape(-1)

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return accuracy, f1


def main():
    # 하이퍼파라미터 설정
    input_size = 24  # MFCC 20개 + Pitch 1개 + F0 1개 + Spectral Flux 1개 + Spectral Entropy 1개 (총 24개)
    hidden_size = 128
    output_size = 2  # 가해자(1), 피해자(0)
    num_layers = 1
    learning_rate = 0.001
    num_epochs = 10
    batch_size = 16
    max_length = 500  # 시퀀스 최대 길이

    # 디바이스 설정 (GPU 사용 여부)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'사용 중인 디바이스: {device}')

    # 데이터 디렉토리 설정
    wav_dir_train = 'C:/Workspace-DoHyeonLim/PythonWorkspace/Lieon-ai/Dataset/Train/Audio'
    label_dir_train = 'C:/Workspace-DoHyeonLim/PythonWorkspace/Lieon-ai/Dataset/Train/Label'

    wav_dir_val = 'C:/Workspace-DoHyeonLim/PythonWorkspace/Lieon-ai/Dataset/Val/Audio'
    label_dir_val = 'C:/Workspace-DoHyeonLim/PythonWorkspace/Lieon-ai/Dataset/Val/Label'

    wav_dir_test = 'C:/Workspace-DoHyeonLim/PythonWorkspace/Lieon-ai/Dataset/Test/Audio'
    label_dir_test = 'C:/Workspace-DoHyeonLim/PythonWorkspace/Lieon-ai/Dataset/Test/Label'

    # 데이터 로드
    wav_paths_train, label_paths_train = get_file_paths(wav_dir_train, label_dir_train)
    wav_paths_val, label_paths_val = get_file_paths(wav_dir_val, label_dir_val)
    wav_paths_test, label_paths_test = get_file_paths(wav_dir_test, label_dir_test)

    # DataLoader 생성
    train_loader = create_dataloader(
        wav_path=wav_paths_train,
        label_path=label_paths_train,
        max_length=max_length,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = create_dataloader(
        wav_path=wav_paths_val,
        label_path=label_paths_val,
        max_length=max_length,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = create_dataloader(
        wav_path=wav_paths_test,
        label_path=label_paths_test,
        max_length=max_length,
        batch_size=batch_size,
        shuffle=False
    )

    # 모델 초기화 및 디바이스 이동
    model = ESN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        nonlinearity='tanh',
        batch_first=True,
        leaking_rate=1.0,
        spectral_radius=0.9,
        w_ih_scale=1.0,
        lambda_reg=0.0,
        density=1.0,
        w_io=False,
        readout_training='gd',
        output_steps='all'
    ).to(device)

    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")

    # 손실 함수 및 최적화기 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 학습 및 검증
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch + 1}/{num_epochs}], ')
        train_loss, train_accuracy, train_f1 = train(model, train_loader, criterion, optimizer, device)
        val_accuracy, val_f1 = evaluate(model, val_loader, device)

        print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Training F1 Score: {train_f1:.4f},'
              f'Validation Accuracy: {val_accuracy:.4f}, Validation F1 Score: {val_f1:.4f}')

    # 테스트
    test_accuracy, test_f1 = evaluate(model, test_loader, device)
    print(f'Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}')

    # 모델 저장
    model_path = 'esn_model_train3.pth'
    torch.save(model.state_dict(), model_path)
    print(f'The model saved as {model_path}.')

if __name__ == '__main__':
    main()
