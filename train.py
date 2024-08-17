import torch
import os
from sklearn.metrics import f1_score, accuracy_score
from Preprocessing.preprocessing import create_dataloader
from Classifier.utils.utils import prepare_target
from torch.utils.data import DataLoader
from Classifier.nn import ESN
import Classifier.nn.esn as esn


def load_data_labels_path(wav_dir, label_dir):
    wav_path = [os.path.join(wav_dir, file) for file in os.listdir(wav_dir) if file.endswith('.wav')]
    label_path = [os.path.join(label_dir, file) for file in os.listdir(label_dir) if file.endswith('.csv')]
    return wav_path, label_path


def evaluate_model(esn, dataloader, washout, criterion, device):
    esn.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            washout_tensor = torch.tensor([washout] * inputs.size(0)).to(device)
            outputs, _ = esn(inputs, washout_tensor)

            outputs = outputs[:, washout:, :]
            targets = targets[:, washout:]

            min_length = min(outputs.size(1), targets.size(1))
            outputs = outputs[:, :min_length, :].squeeze(-1)
            targets = targets[:, :min_length]

            loss = criterion(outputs, targets)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, -1)
            correct += (predicted == targets).sum().item()
            total += targets.numel()

    average_loss = total_loss / len(dataloader)
    accuracy = correct / total
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    return average_loss, accuracy, f1


def train(esn, train_loader, val_loader, num_epochs, washout, criterion, optimizer, device):
    best_loss = float('inf')
    best_model = None
    esn.to(device)

    for epoch in range(num_epochs):
        esn.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            washout_tensor = torch.tensor([washout] * inputs.size(0)).to(device)
            outputs, _ = esn(inputs, washout_tensor)

            outputs = outputs[:, washout:, :]
            targets = targets[:, washout:]

            min_length = min(outputs.size(1), targets.size(1))
            outputs = outputs[:, :min_length, :].squeeze(-1)
            targets = targets[:, :min_length]

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, -1)
            correct += (predicted == targets).sum().item()
            total += targets.numel()

        train_loss = total_loss / len(train_loader)
        train_accuracy = correct / total
        val_loss, val_accuracy, val_f1 = evaluate_model(esn, val_loader, washout, criterion, device)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = esn.state_dict()
            torch.save(best_model, 'esn_model_best.pth')

    if best_model is None:
        best_model = esn.state_dict()
        torch.save(best_model, 'esn_model_best.pth')

    print("Training complete. Best model saved.")
    return best_model


def test(esn, test_loader, washout, criterion, device):
    test_loss, test_accuracy, test_f1 = evaluate_model(esn, test_loader, washout, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}')


def main():
    wav_dir_train = '/Users/imdohyeon/Documents/PythonWorkspace/Lieon-ai/dataset/Train/Audio'
    wav_dir_val = '/Users/imdohyeon/Documents/PythonWorkspace/Lieon-ai/dataset/Val/Audio'
    wav_dir_test = '/Users/imdohyeon/Documents/PythonWorkspace/Lieon-ai/dataset/Test/Audio'
    label_dir_train = '/Users/imdohyeon/Documents/PythonWorkspace/Lieon-ai/dataset/Train/Label'
    label_dir_val = '/Users/imdohyeon/Documents/PythonWorkspace/Lieon-ai/dataset/Val/Label'
    label_dir_test = '/Users/imdohyeon/Documents/PythonWorkspace/Lieon-ai/dataset/Test/Label'

    max_length = 200  # Set maximum length for padding/truncating
    batch_size = 16
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize ESN
    esn = ESN(input_size=input_size, hidden_size=hidden_size, output_size=output_size,
              num_layers=num_layers, leaking_rate=leaking_rate, spectral_radius=spectral_radius,
              lambda_reg=lambda_reg, w_ih_scale=w_ih_scale, density=density, readout_training=readout_training,
              output_steps=output_steps, batch_first=True)

    # Training loop
    num_epochs = 50
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Load data path and labels path
    wav_path_train = load_data_labels_path(wav_dir_train, label_dir_train)
    wav_path_val = load_data_labels_path(wav_dir_val, label_dir_val)
    wav_path_test = load_data_labels_path(wav_dir_test, label_dir_test)

    # Create DataLoader
    train_dataloader = create_dataloader(wav_path_train, max_length, batch_size, shuffle=True)
    val_dataloader = create_dataloader(wav_path_val, max_length, batch_size, shuffle=False)
    test_dataloader = create_dataloader(wav_path_test, max_length, batch_size, shuffle=False)

    train(esn, train_dataloader, val_dataloader, num_epochs, washout, criterion, optimizer, device)

    # Load the best model for testing
    esn.load_state_dict(torch.load('esn_model_001.pth'))
    test(esn, test_dataloader, washout, criterion, device)

    print('Training complete.')


if __main__ == '__main__':
    main()