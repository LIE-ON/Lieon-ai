# train.py - script mode

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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


def main(args):
    # 디바이스 설정 (GPU 사용 여부)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'사용 중인 디바이스: {device}')

    # 데이터 로드 (SageMaker의 경로에서 데이터 로드)
    wav_paths_train, label_paths_train = get_file_paths(args.train_dir, args.label_dir)
    wav_paths_val, label_paths_val = get_file_paths(args.val_dir, args.val_label_dir)
    wav_paths_test, label_paths_test = get_file_paths(args.test_dir, args.test_label_dir)

    # DataLoader 생성
    train_loader = create_dataloader(
        wav_path=wav_paths_train,
        label_path=label_paths_train,
        max_length=args.max_length,
        batch_size=args.batch_size,
        shuffle=True
    )

    val_loader = create_dataloader(
        wav_path=wav_paths_val,
        label_path=label_paths_val,
        max_length=args.max_length,
        batch_size=args.batch_size,
        shuffle=False
    )

    # 모델 초기화 및 디바이스 이동
    model = ESN(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        output_size=args.output_size,
        num_layers=args.num_layers,
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

    # 손실 함수 및 최적화기 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 학습 및 검증
    for epoch in range(args.num_epochs):
        print(f'Epoch [{epoch + 1}/{args.num_epochs}], ')
        train_loss, train_accuracy, train_f1 = train(model, train_loader, criterion, optimizer, device)
        val_accuracy, val_f1 = evaluate(model, val_loader, device)

        print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Training F1 Score: {train_f1:.4f},'
              f'Validation Accuracy: {val_accuracy:.4f}, Validation F1 Score: {val_f1:.4f}')

    # 테스트
    test_accuracy, test_f1 = evaluate(model, val_loader, device)
    print(f'Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}')

    # SageMaker가 기대하는 모델 저장 경로에 모델 저장
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'esn_model.pth'))
    print(f'The model saved as {os.path.join(args.model_dir, "esn_model.pth")}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # SageMaker hyperparameters and paths
    parser.add_argument('--input_size', type=int, default=24)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--output_size', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=500)

    parser.add_argument('--train_dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--label_dir', type=str, default=os.environ['SM_CHANNEL_LABEL'])
    parser.add_argument('--val_dir', type=str, default=os.environ['SM_CHANNEL_VAL'])
    parser.add_argument('--val_label_dir', type=str, default=os.environ['SM_CHANNEL_VAL_LABEL'])
    parser.add_argument('--test_dir', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--test_label_dir', type=str, default=os.environ['SM_CHANNEL_TEST_LABEL'])

    # Model directory provided by SageMaker
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])

    args = parser.parse_args()
    main(args)
