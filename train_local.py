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


def evaluate(model, data_loader, device):
    model.eval()

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # ESN 모델의 forward를 통해 출력 얻기
            outputs = model(inputs)

            # 가장 높은 확률을 가진 클래스를 예측
            _, predictions = torch.max(outputs, 1)

            all_targets.extend(targets.cpu().numpy())  # 정답
            all_predictions.extend(predictions.cpu().numpy())  # 예측값

    accuracy = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions, average='weighted')  # 다중 클래스일 경우 weighted 사용

    return accuracy, f1


def main():
    # 하이퍼파라미터 설정
    input_size = 24  # MFCC 20개 + Pitch 1개 + F0 1개 + Spectral Flux 1개 + Spectral Entropy 1개 (총 24개)
    hidden_size = 128
    output_size = 2  # 가해자(1), 피해자(0)
    num_layers = 1
    learning_rate = 0.001
    num_epochs = 20
    batch_size = 64
    max_length = 500  # 시퀀스 최대 길이

    # 디바이스 설정 (GPU 사용 여부)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'사용 중인 디바이스: {device}')

    # 데이터 디렉토리 설정
    wav_dir_train = '/Users/imdohyeon/Documents/PythonWorkspace/Lieon-ai/Dataset/Train/Audio'
    label_dir_train = '/Users/imdohyeon/Documents/PythonWorkspace/Lieon-ai/Dataset/Train/Label'

    wav_dir_val = '/Users/imdohyeon/Documents/PythonWorkspace/Lieon-ai/Dataset/Val/Audio'
    label_dir_val = '/Users/imdohyeon/Documents/PythonWorkspace/Lieon-ai/Dataset/Val/Label'

    wav_dir_test = '/Users/imdohyeon/Documents/PythonWorkspace/Lieon-ai/Dataset/Test/Audio'
    label_dir_test = '/Users/imdohyeon/Documents/PythonWorkspace/Lieon-ai/Dataset/Test/Label'

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

    # ESN 학습을 위한 데이터 준비 (순차 데이터)
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch + 1}/{num_epochs}]')

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # ESN 모델의 fit 메소드를 사용하여 학습
            model.fit(inputs, targets)  # fit을 통해 readout layer 학습

        # Evaluate
        val_accuracy, val_f1 = evaluate(model, val_loader, device)
        print(f'Validation Accuracy: {val_accuracy:.4f}, Validation F1 Score: {val_f1:.4f}')

    # 테스트 데이터 평가
    test_accuracy, test_f1 = evaluate(model, test_loader, device)
    print(f'Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}')

    # 모델 저장
    model_path = 'esn_g4dn2xlarge_50hr.pth'
    torch.save(model.state_dict(), model_path)
    print(f'The model saved as {model_path}.')


if __name__ == '__main__':
    main()
