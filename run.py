# 미완성




# WAV 파일이 있는 디렉토리
wav_dir = 'dataset/train'
wav_files = [os.path.join(wav_dir, file) for file in os.listdir(wav_dir) if file.endswith('.wav')]

# 데이터셋 생성
dataset = WAVDataset(wav_files)

# 데이터셋을 트레인 셋과 테스트 셋으로 분할
train_dataset, test_dataset = split_dataset(wav_dataset, train_ratio=0.8)

# 데이터로더 생성
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


# 여기서부터 모델 학습 및 예측 코드

# ESN 모델 파라미터 정의
input_size = train_dataset[0].shape[1] * train_dataset[0].shape[2]  # 피처 수
hidden_size = 100  # 예시 은닉층 크기
output_size = 1  # 예시 출력 크기
washout = 0  # 예시 워시오트 기간

# ESN 모델 생성
model = ESN(input_size, hidden_size, output_size)

# 학습을 위한 타겟 데이터 준비 (타겟 데이터가 있는 경우)
# 여기서는 단순히 예시로 랜덤 타겟 데이터를 생성
target_data = torch.randn(len(train_dataset), output_size)
flat_target = prepare_target(target_data.unsqueeze(1), [len(train_dataset)], [washout])

# 모델 학습
for batch in train_dataloader:
    batch = batch.view(batch.size(0), -1)  # 배치 차원 추가 및 reshape
    output, _ = model(batch.unsqueeze(1), [washout], None, flat_target)
    model.fit()

# 예측
for batch in test_dataloader:
    batch = batch.view(batch.size(0), -1)
    output, _ = model(batch.unsqueeze(1), [washout], None)
    print(output)
    break
