import torch
# from Preprocessing.preprocessing import create_dataloader
from torch.utils.data import DataLoader
from Preprocessing.preprocessed_dataset import PreprocessedDataset
from sklearn.metrics import accuracy_score, f1_score
from Classifier.nn.esn import ESN
import time
import boto3
import os


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


"""
def get_file_paths(wav_dir, label_dir):
    s3 = boto3.client('s3')

    # S3 경로에서 버킷명과 경로를 분리
    wav_dir = wav_dir.replace("s3://", "")
    label_dir = label_dir.replace("s3://", "")

    wav_bucket, wav_prefix = wav_dir.split('/', 1)
    label_bucket, label_prefix = label_dir.split('/', 1)

    # WAV 파일 목록 가져오기
    wav_files = []
    response = s3.list_objects_v2(Bucket=wav_bucket, Prefix=wav_prefix)
    for obj in response.get('Contents', []):
        if obj['Key'].endswith('.wav'):
            wav_files.append(f"s3://{wav_bucket}/{obj['Key']}")

    # 라벨 파일 목록 가져오기
    label_files = []
    response = s3.list_objects_v2(Bucket=label_bucket, Prefix=label_prefix)
    for obj in response.get('Contents', []):
        if obj['Key'].endswith('.csv'):
            label_files.append(f"s3://{label_bucket}/{obj['Key']}")

    return wav_files, label_files
"""


def evaluate(model, washout_rate, data_loader, device):
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            seq_len = inputs.size(1)
            washout_length = int(washout_rate * seq_len)
            washout_list = [washout_length] * inputs.size(0)

            # 모델의 출력 얻기
            outputs, _ = model(inputs, washout_list)

            # 타겟을 출력과 맞추기 위해 자르기
            trimmed_targets = targets[:, washout_length:]

            # 평균값 계산 후 예측 클래스 선택
            mean_outputs = torch.mean(outputs, dim=1)
            predicted = mean_outputs.argmax(dim=1)
            true_labels = trimmed_targets[:, -1]  # 마지막 라벨 사용

            all_predictions.extend(predicted.cpu().detach().numpy())
            all_targets.extend(true_labels.cpu().detach().numpy())

    # 정확도와 F1 스코어 계산
    accuracy = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    return accuracy, f1


def main():
    # 데이터 로드
    """
    wav_paths_train, label_paths_train = get_file_paths(wav_dir_train, label_dir_train)
    wav_paths_val, label_paths_val = get_file_paths(wav_dir_val, label_dir_val)
    wav_paths_test, label_paths_test = get_file_paths(wav_dir_test, label_dir_test)

    # DataLoader 생성
    train_loader = create_dataloader(
        wav_path=wav_paths_train, label_path=label_paths_train, max_length=max_length,
        batch_size=batch_size, shuffle=True)

    val_loader = create_dataloader(
        wav_path=wav_paths_val, label_path=label_paths_val, max_length=max_length,
        batch_size=batch_size, shuffle=False)

    test_loader = create_dataloader(
        wav_path=wav_paths_test, label_path=label_paths_test, max_length=max_length,
        batch_size=batch_size, shuffle=False)
    """
    wav_path_train = download_s3_file(wav_dir_train)
    wav_path_val = download_s3_file(wav_dir_val)
    wav_path_test = download_s3_file(wav_dir_test)
    
    train_dataset = PreprocessedDataset(wav_path_train)
    val_dataset = PreprocessedDataset(wav_path_val)
    test_dataset = PreprocessedDataset(wav_path_test)
    
    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # 모델 초기화 및 디바이스 이동
    model = ESN(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers,
        batch_first=True,
        w_io=False,
        leaking_rate=1.0, spectral_radius=0.9, w_ih_scale=1.0, lambda_reg=0.0, density=1.0,
        nonlinearity='tanh', readout_training='svd', output_steps='all'
    ).to(device)

    # ESN 학습을 위한 데이터 준비 (순차 데이터)
    for epoch in range(num_epochs):
        start = time.time()
        print(f'Epoch [{epoch + 1}/{num_epochs}]')

        # Reset the matrices used for training the readout layer
        model.XTX = None
        model.XTy = None

        # Training loop: update the internal states of the model using the training data
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            seq_len = inputs.size(1)
            washout_list = [int(washout_rate * seq_len)] * inputs.size(0)

            # Prepare the targets by trimming the washout period
            trimmed_targets = targets[:, washout_list[0]:]

            # Reshape targets to match the expected dimensions
            target_sequence = trimmed_targets.reshape(-1, 1).float()

            # Collect data for readout training (without updating weights yet)
            model(inputs, washout_list, None, target_sequence)

        # Fit the readout layer once after processing all batches in the epoch
        if epoch == num_epochs - 1:  # Only perform readout training after the last epoch
            model.fit()

        val_accuracy, val_f1 = evaluate(model, washout_rate, val_loader, device)
        print(f'Validation Accuracy: {val_accuracy:.4f}, Validation F1 Score: {val_f1:.4f}')

        print("Epoch", epoch + 1, ": Ended in", time.time() - start, "seconds.")

    # 테스트 데이터 평가
    test_accuracy, test_f1 = evaluate(model, washout_rate, test_loader, device)
    print(f'Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}')

    # 모델 저장
    model_path = 'esn_model_test1111.pth'
    torch.save(model.state_dict(), model_path)
    print(f'The model saved as {model_path}.')


if __name__ == '__main__':
    # 디바이스 설정 (GPU 사용 여부)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # 하이퍼파라미터 설정
    input_size = 24  # MFCC 20개 + Pitch 1개 + F0 1개 + Spectral Flux 1개 + Spectral Entropy 1개 (총 24개)
    hidden_size = 850
    output_size = 2  # 가해자(1), 피해자(0)
    washout_rate = 0.2
    num_layers = 1
    num_epochs = 10
    batch_size =2048
    max_length = 850  # 시퀀스 최대 길이

    # SageMaker에서 제공하는 데이터 경로 사용
    wav_dir_train = 's3://lieon-data/Dataset/Train/preprocessed_train_data.pt'
    wav_dir_val = 's3://lieon-data/Dataset/Val/preprocessed_val_data.pt'
    wav_dir_test = 's3://lieon-data/Dataset/Test/preprocessed_test_data.pt'

    main()