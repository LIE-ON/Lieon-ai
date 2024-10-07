import torch
from Preprocessing.preprocessing import create_dataloader
from Classifier.nn.esn import ESN
import time
import boto3


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


def accuracy_correct(y_pred, y_true):
    labels = torch.argmax(y_pred, 1).type(y_pred.type())
    correct = len((labels == y_true).nonzero())
    return correct


def evaluate(model, washout_rate, data_loader, device):
    model.eval()

    all_targets = 0
    all_predictions = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            washout_list = [int(washout_rate * inputs.size(0))] * inputs.size(1)

            # ESN 모델의 forward를 통해 출력 얻기
            outputs, hidden = model(inputs, washout_list)

            # 가장 높은 확률을 가진 클래스를 예측
            _, predictions = torch.max(outputs, 1)

            all_targets += inputs.size(1)
            all_predictions += loss_fcn(outputs[-1], targets.type(torch.get_default_dtype()))

    eval_acc = all_predictions / all_targets
    return eval_acc


def main():
    # 데이터 로드
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

    # 모델 초기화 및 디바이스 이동
    model = ESN(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, nonlinearity='tanh', batch_first=True,
        leaking_rate=1.0, spectral_radius=0.9, w_ih_scale=1.0,
        lambda_reg=0.0, density=1.0, w_io=False,
        readout_training='gd', output_steps='all'
    ).to(device)

    # ESN 학습을 위한 데이터 준비 (순차 데이터)
    for epoch in range(num_epochs):
        start = time.time()
        print(f'Epoch [{epoch + 1}/{num_epochs}]')

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            washout_list = [int(washout_rate * inputs.size(0))] * inputs.size(1)

            # ESN 모델의 fit 메소드를 사용하여 학습
            model(inputs, washout_list, None, targets)
            model.fit()  # fit을 통해 readout layer 학습

        # Evaluate
        val_accuracy = evaluate(model, washout_rate, val_loader, device)
        print(f'Validation Accuracy: {val_accuracy:.4f}')

        print("Epoch", epoch + 1, ": Ended in", time.time() - start, "seconds.")

    # 테스트 데이터 평가
    test_accuracy = evaluate(model, washout_rate, test_loader, device)
    print(f'Test Accuracy: {test_accuracy:.4f}')

    # 모델 저장
    model_path = '/opt/ml/model/esn_model_g4dn2xlarge_50hr.pth'
    torch.save(model.state_dict(), model_path)
    print(f'The model saved as {model_path}.')


if __name__ == '__main__':
    # 디바이스 설정 (GPU 사용 여부)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # 하이퍼파라미터 설정
    input_size = 24  # MFCC 20개 + Pitch 1개 + F0 1개 + Spectral Flux 1개 + Spectral Entropy 1개 (총 24개)
    hidden_size = 500
    output_size = 2  # 가해자(1), 피해자(0)
    washout_rate = 0.2
    num_layers = 1
    num_epochs = 10
    batch_size = 64
    max_length = 500  # 시퀀스 최대 길이
    loss_fcn = accuracy_correct

    # SageMaker에서 제공하는 데이터 경로 사용
    wav_dir_train = 's3://lieon-data/Dataset/Train/Audio'
    label_dir_train = 's3://lieon-data/Dataset/Train/Label'

    wav_dir_val = 's3://lieon-data/Dataset/Val/Audio'
    label_dir_val = 's3://lieon-data/Dataset/Val/Label'

    wav_dir_test = 's3://lieon-data/Dataset/Test/Audio'
    label_dir_test = 's3://lieon-data/Dataset/Test/Label'

    main()
