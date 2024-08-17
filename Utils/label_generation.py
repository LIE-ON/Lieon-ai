import os
import pandas as pd
import torch
# from concurrent.futures import ThreadPoolExecutor, as_completed
from pyannote.audio import Pipeline


def speaker_diarization(audio_path, output_csv_path, use_auth_token):
    """
    오디오 파일에 대한 speaker diarization 수행
    :param audio_path: 오디오 파일 경로
    :param output_csv_path: 출력 CSV 파일 경로
    :param use_auth_token: Hugging Face API 토큰
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=use_auth_token)
    pipeline = pipeline.to(device)

    diarization = pipeline(audio_path, num_speakers=2)

    label_mapping = {}
    label_counter = 1

    output_dir = os.path.dirname(output_csv_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Start', 'End', 'Speaker', 'Label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in label_mapping:
                label_mapping[speaker] = label_counter
                label_counter += 1
            label = label_mapping[speaker]
            writer.writerow({'Start': turn.start, 'End': turn.end, 'Speaker': speaker, 'Label': label})



def process_single_file(audio_path, output_csv_dir, use_auth_token):
    """
    개별 파일에 대한 speaker diarization 실행 함수
    """
    if not os.path.exists(audio_path):
        print(f"File {audio_path} does not exist. Skipping...")
        return

    try:
        # label 파일명을 생성
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        label_number = ''.join(filter(str.isdigit, base_name))  # 숫자 부분 추출
        output_csv_path = os.path.join(output_csv_dir, f'label{label_number}.csv')

        # speaker_diarization 호출
        diarization = speaker_diarization(audio_path, output_csv_path, use_auth_token)

        # pandas를 사용하여 CSV 작성
        records = []
        label_mapping = {}
        label_counter = 1

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in label_mapping:
                label_mapping[speaker] = label_counter
                label_counter += 1
            label = label_mapping[speaker]
            records.append({
                'Start': turn.start,
                'End': turn.end,
                'Speaker': speaker,
                'Label': label
            })

        df = pd.DataFrame.from_records(records)
        df.to_csv(output_csv_path, index=False)

    except Exception as e:
        print(f"Error processing file {audio_path}: {e}")


def process_audio_dirs_in_parallel(audio_dir, output_csv_dir, use_auth_token, max_workers=4):
    """
    병렬 처리를 이용한 오디오 파일 처리
    :param audio_dir: 입력 오디오 파일 디렉토리
    :param output_csv_dir: 출력 파일 디렉토리
    :param use_auth_token: Hugging Face API 토큰
    :param max_workers: 병렬 처리를 위한 최대 worker 수
    """
    audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_file, audio_file, output_csv_dir, use_auth_token): audio_file for audio_file in audio_files}
        for future in as_completed(futures):
            audio_file = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing file {audio_file}: {e}")

    # 모든 작업이 완료되면 종료 신호 출력
    print("All files have been processed.")


def process_audio_files_sequentially(audio_dir, output_csv_dir, use_auth_token):
    """
    오디오 파일을 순차적으로 처리
    :param audio_dir: 입력 오디오 파일 디렉토리
    :param output_csv_dir: 출력 파일 디렉토리
    :param use_auth_token: Hugging Face API 토큰
    """
    audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]

    for audio_file in audio_files:
        process_single_file(audio_file, output_csv_dir, use_auth_token)

    print("All files have been processed.")


def main():
    base_dir = '/Users/imdohyeon/Library/CloudStorage/GoogleDrive-dhlim1598@gmail.com/.shortcut-targets-by-id/1GKf6cKNuFHdu7j8BO1RrBifSerASlNG8/LIEON_DATA/[Temp]total_data/'
    audio_dir = os.path.join(base_dir, "Audio")
    output_csv_dir = os.path.join(base_dir, "Label")

    # Hugging Face API 토큰
    use_auth_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # process_audio_dirs_in_parallel(audio_dir, output_csv_dir, use_auth_token, max_workers=4)
    process_audio_files_sequentially(audio_dir, output_csv_dir, use_auth_token)

    # Not completed yet

if __name__ == "__main__":
    main()
