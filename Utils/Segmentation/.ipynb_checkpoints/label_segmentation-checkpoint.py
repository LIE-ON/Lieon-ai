import os
import pandas as pd
from pydub import AudioSegment

# 10초 기준의 초 길이
CHUNK_DURATION = 10


def create_output_dir(input_dir):
    output_dir = f"{input_dir}_edit"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def get_audio_duration(file_path):
    # pydub을 이용해 오디오 파일의 길이를 읽음 (초 단위로 변환)
    audio = AudioSegment.from_file(file_path)
    return len(audio) / 1000  # ms 단위를 초로 변환


def split_labels(label_file_path, audio_duration, output_dir):
    # 라벨 파일을 읽기
    labels = pd.read_csv(label_file_path)
    file_name = os.path.basename(label_file_path)
    file_stem, ext = os.path.splitext(file_name)

    # 오디오 파일의 길이에 맞춰 분할
    num_chunks = int(audio_duration // CHUNK_DURATION) + 1

    # 각 chunk(10초 단위)로 분할하여 처리
    for i in range(num_chunks):
        start_time = i * CHUNK_DURATION
        end_time = (i + 1) * CHUNK_DURATION

        # 현재 chunk에 해당하는 라벨 데이터 필터링
        chunk_data = labels[(labels['End'] > start_time) & (labels['Start'] < end_time)].copy()

        # 현재 chunk에서의 상대적 시간으로 변환 (0초부터 시작)
        chunk_data['Start'] = chunk_data['Start'].clip(lower=start_time) - start_time
        chunk_data['End'] = chunk_data['End'].clip(upper=end_time) - start_time

        # 마지막 구간이 10초가 안 되는 경우 0으로 채움
        if i == num_chunks - 1 and audio_duration % CHUNK_DURATION != 0:
            last_chunk_duration = audio_duration % CHUNK_DURATION
            chunk_data = chunk_data[chunk_data['End'] <= last_chunk_duration]  # 남은 부분만 포함
            chunk_data.loc[len(chunk_data)] = [last_chunk_duration, CHUNK_DURATION, "None", 0]  # 0으로 채우기

        # 파일로 저장
        chunk_name = f"{file_stem}_{i + 1}{ext}"
        chunk_data.to_csv(os.path.join(output_dir, chunk_name), index=False)


def process_directory(audio_dir, label_dir):
    # 오디오 파일과 라벨 파일의 output 디렉토리 생성
    output_dir = create_output_dir(label_dir)

    # 라벨 파일 리스트 가져오기
    label_files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.csv')]

    # 각 라벨 파일에 대응하는 오디오 파일을 찾아서 처리
    for label_file in label_files:
        # 대응하는 오디오 파일 경로 구하기
        file_name = os.path.basename(label_file).replace('label', 'data').replace('.csv', '.wav')
        audio_file_path = os.path.join(audio_dir, file_name)

        if os.path.exists(audio_file_path):
            # 오디오 파일 길이 가져오기
            audio_duration = get_audio_duration(audio_file_path)
            # 라벨 파일을 오디오 파일 길이에 맞춰 분할
            split_labels(label_file, audio_duration, output_dir)
        else:
            print(f"오디오 파일을 찾을 수 없습니다: {audio_file_path}")


if __name__ == "__main__":
    train_audio_directory = "/Users/imdohyeon/Documents/PythonWorkspace/Lieon-ai/Dataset/Train/Audio"
    train_label_directory = "/Users/imdohyeon/Documents/PythonWorkspace/Lieon-ai/Dataset/Train/Label"

    val_audio_directory = "/Users/imdohyeon/Documents/PythonWorkspace/Lieon-ai/Dataset/Val/Audio"
    val_label_directory = "/Users/imdohyeon/Documents/PythonWorkspace/Lieon-ai/Dataset/Val/Label"

    test_audio_directory = "/Users/imdohyeon/Documents/PythonWorkspace/Lieon-ai/Dataset/Test/Audio"
    test_label_directory = "/Users/imdohyeon/Documents/PythonWorkspace/Lieon-ai/Dataset/Test/Label"

    process_directory(train_audio_directory, train_label_directory)
    process_directory(val_audio_directory, val_label_directory)
    process_directory(test_audio_directory, test_label_directory)
