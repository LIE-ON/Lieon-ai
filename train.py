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
    mean_predictions = torch.mean(y_pred, dim=1)
    labels = (mean_predictions >= 0.5).long()  # 평균값이 0.5보다 크면 사기범(1)으로 간주, 그렇지 않으면 수신자(0)로 간주
    correct = (labels == y_true).sum().item()  # 예측이 맞은 개수

    return correct


def evaluate(model, washout_rate, data_loader, device):
    model.eval()
    all_targets = 0
    all_correct = 0  # 변수명 변경

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            seq_len = inputs.size(1)
            washout_length = int(washout_rate * seq_len)

            # washout_list 생성
            washout_list = [washout_length] * inputs.size(0)

            # 모델 출력 얻기
            outputs, hidden = model(inputs, washout_list)

            # targets의 앞부분을 제거하여 outputs와 길이 맞추기
            trimmed_targets = targets[:, washout_length:]

            # 출력과 라벨의 크기 확인
            assert outputs.size(1) == trimmed_targets.size(1), \
                f"Outputs and targets have mismatched sizes: {outputs.size(1)} vs {trimmed_targets.size(1)}"

            # 시퀀스 단위의 평균을 계산하여 예측 및 라벨 생성
            mean_outputs = torch.mean(outputs, dim=1)
            mean_targets = torch.mean(trimmed_targets.float(), dim=1).long()

            # 정확도 계산
            batch_correct = accuracy_correct(mean_outputs.argmax(dim=1), mean_targets)
            batch_size = inputs.size(0)
            all_targets += batch_size
            all_correct += batch_correct * batch_size

    eval_acc = all_correct / all_targets
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
        readout_training='gd', output_steps='mean'
    ).to(device)

    # ESN 학습을 위한 데이터 준비 (순차 데이터)
    for epoch in range(num_epochs):
        start = time.time()
        print(f'Epoch [{epoch + 1}/{num_epochs}]')

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")  # debugging code
            washout_list = [int(washout_rate * inputs.size(0))] * inputs.size(1)

            # ESN 모델의 fit 메소드를 사용하여 학습
            model(inputs, washout_list, None, targets)
            model.fit()  # fit을 통해 readout layer 학습

            # fit 실행 횟수 증가 및 출력
            print(f'--- Fit iteration [{batch_idx + 1}/{len(train_loader)}]')

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
    hidden_size = 850
    output_size = 2  # 가해자(1), 피해자(0)
    washout_rate = 0.2
    num_layers = 1
    num_epochs = 10
    batch_size =2048
    max_length = 850  # 시퀀스 최대 길이
    loss_fcn = accuracy_correct

    # SageMaker에서 제공하는 데이터 경로 사용
    wav_dir_train = 's3://lieon-data/Dataset/Train/Audio'
    label_dir_train = 's3://lieon-data/Dataset/Train/Label'

    wav_dir_val = 's3://lieon-data/Dataset/Val/Audio'
    label_dir_val = 's3://lieon-data/Dataset/Val/Label'

    wav_dir_test = 's3://lieon-data/Dataset/Test/Audio'
    label_dir_test = 's3://lieon-data/Dataset/Test/Label'

    main()
