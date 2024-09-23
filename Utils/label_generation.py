import os
import torch
import csv
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
import concurrent.futures
import re


def load_pipeline(use_auth_token):
    """
    PyAnnote 파이프라인을 로드하는 함수
    """
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=use_auth_token
    )
    return pipeline


def speaker_diarization(audio_path, output_csv_path, pipeline):
    """
    주어진 오디오 파일에 대해 speaker diarization 수행 및 CSV로 저장
    """
    # Diarization 수행
    diarization = pipeline(audio_path, num_speakers=2)

    # 라벨링 결과 출력 및 CSV 파일 저장
    label_mapping = {}
    label_counter = 0  # 라벨 카운터를 1로 시작하여 묵음 라벨과 겹치지 않도록 함

    # CSV 파일 열기 및 헤더 작성
    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Start', 'End', 'Speaker', 'Label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        # 화자 구간 라벨링
        annotation = Annotation()
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in label_mapping:
                label_mapping[speaker] = label_counter
                label_counter += 1
            label = label_mapping[speaker]
            writer.writerow({'Start': turn.start, 'End': turn.end, 'Speaker': speaker, 'Label': label})
            annotation[Segment(start=turn.start, end=turn.end)] = speaker


def process_file(audio_file, output_csv_dir, pipeline):
    """
    각 파일에 대한 diarization 수행 및 결과 저장
    """
    # 파일 이름에서 숫자 추출 (예: data1.wav -> 1)
    match = re.search(r'\d+', os.path.basename(audio_file))
    if match:
        file_number = match.group(0)
        # 대응되는 label 파일 이름 설정 (label1.csv, label2.csv, ...)
        output_csv_path = os.path.join(output_csv_dir, f"label{file_number}.csv")
        speaker_diarization(audio_file, output_csv_path, pipeline)


def process_file_for_augment(audio_file, output_csv_dir, pipeline):
    """
    각 파일에 대한 diarization 수행 및 결과 저장
    """
    # 파일 이름에서 필요한 부분 추출 (예: data1.wav_augment1.wav -> label1.wav_augment1.csv)
    base_name = os.path.basename(audio_file)
    base_name_without_ext = os.path.splitext(base_name)[0]  # 확장자를 제거한 파일명

    # 대응되는 label 파일 이름 설정
    output_csv_path = os.path.join(output_csv_dir, f"label{base_name_without_ext}.csv")

    speaker_diarization(audio_file, output_csv_path, pipeline)


def main():
    base_dir = "C:/Workspace-DoHyeonLim/PythonWorkspace/Lieon-ai/Dataset/[Temp]total_data/Augmented/"
    audio_dir = os.path.join(base_dir, "Audio/")
    output_csv_dir = os.path.join(base_dir, "Label/")

    # 오디오 파일 목록 가져오기
    audio_files = [os.path.join(audio_dir, file) for file in os.listdir(audio_dir) if file.endswith('.wav')]

    use_auth_token = os.environ["HUGGINGFACEHUB_API_TOKEN"]

    # 순차적으로 파일 처리
    for idx, audio_file in enumerate(audio_files, start=1):
        print(f'Processing: {audio_file} ({idx}/{len(audio_files)})')  # 진행 상태 표시
        # PyAnnote pipeline 로드
        pipeline = load_pipeline(use_auth_token)

        try:
            # process_file(audio_file, output_csv_dir, pipeline)
            process_file_for_augment(audio_file, output_csv_dir, pipeline)
        except Exception as e:
            print(f"Error processing file {audio_file}: {e}")


if __name__ == "__main__":
    main()
