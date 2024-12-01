import torch
import boto3
from torch.utils.data import DataLoader
from Preprocessing.preprocessing import WAVDataset, download_s3_file
from tqdm import tqdm  # Import tqdm for progress bars


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


def preprocess_and_save(wav_paths, label_paths, max_length, save_path):
    dataset = WAVDataset(wav_paths, label_paths, max_length)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=False)

    features = []
    labels = []

    # Initialize tqdm progress bar
    for idx, (X, y) in enumerate(tqdm(dataloader, desc='Preprocessing', unit='file')):
        # Remove batch dimension from features and labels
        X = X.squeeze(0)  # Shape: [seq_len, feature_dim]
        y = y.squeeze(0)  # Shape: [label_dim]

        features.append(X)
        labels.append(y)

    # Stack features and labels
    features = torch.stack(features)  # Shape: [num_samples, seq_len, feature_dim]
    labels = torch.stack(labels)      # Shape: [num_samples, label_dim]

    # Save preprocessed data
    torch.save({'features': features, 'labels': labels}, save_path)
    print(f'Preprocessed data saved to {save_path}')


if __name__ == '__main__':
    # Define your paths and parameters
    wav_dir_train = 's3://lieon-data/Dataset/Train/Audio_edit'
    label_dir_train = 's3://lieon-data/Dataset/Train/Label_edit'

    wav_dir_val = 's3://lieon-data/Dataset/Val/Audio_edit'
    label_dir_val = 's3://lieon-data/Dataset/Val/Label_edit'

    wav_dir_test = 's3://lieon-data/Dataset/Test/Audio_edit'
    label_dir_test = 's3://lieon-data/Dataset/Test/Label_edit'

    wav_paths_train, label_paths_train = get_file_paths(wav_dir_train, label_dir_train)
    wav_paths_val, label_paths_val = get_file_paths(wav_dir_val, label_dir_val)
    wav_paths_test, label_paths_test = get_file_paths(wav_dir_test, label_dir_test)

    save_path_train = 'preprocessed_train_data_ver2.pt'
    save_path_val = 'preprocessed_val_data_ver2.pt'
    save_path_test = 'preprocessed_test_data_ver2.pt'
    max_length = 850  # Or your specific value

    print("Starting preprocessing of training data...")
    preprocess_and_save(wav_paths_train, label_paths_train, max_length, save_path_train)

    print("Starting preprocessing of validation data...")
    preprocess_and_save(wav_paths_val, label_paths_val, max_length, save_path_val)

    print("Starting preprocessing of test data...")
    preprocess_and_save(wav_paths_test, label_paths_test, max_length, save_path_test)
