import os
import logging
import pandas as pd
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from pyannote.audio import Pipeline


def setup_logging():
    """
    로깅 설정을 수행하는 함수
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )


def load_pipeline(use_auth_token, device):
    """
    PyAnnote 파이프라인을 로드하는 함수
    """
    logging.info("Loading PyAnnote pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=use_auth_token
    )
    pipeline.to(device)
    logging.info("Pipeline loaded successfully.")
    return pipeline


def speaker_diarization(audio_path, output_csv_path, pipeline):
    """
    오디오 파일에 대한 speaker diarization 수행
    """
    logging.info(f"Processing file: {audio_path}")

    # Diarization 수행
    diarization = pipeline(audio_path, num_speakers=2)

    # 결과를 데이터프레임으로 변환
    records = []
    label_mapping = {}
    label_counter = 1

    for segment, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in label_mapping:
            label_mapping[speaker] = label_counter
            label_counter += 1
        label = label_mapping[speaker]
        records.append({
            'Start': segment.start,
            'End': segment.end,
            'Speaker': speaker,
            'Label': label
        })

    df = pd.DataFrame(records)

    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # 결과를 CSV로 저장
    df.to_csv(output_csv_path, index=False)
    logging.info(f"Finished processing {audio_path}. Output saved to {output_csv_path}.")


def process_single_file(args):
    """
    개별 파일을 처리하는 함수
    """
    audio_path, output_csv_dir, pipeline = args
    try:
        if not os.path.exists(audio_path):
            logging.warning(f"File {audio_path} does not exist. Skipping...")
            return

        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        label_number = ''.join(filter(str.isdigit, base_name))
        output_csv_path = os.path.join(output_csv_dir, f'label{label_number}.csv')

        speaker_diarization(audio_path, output_csv_path, pipeline)
    except Exception as e:
        logging.error(f"Error processing file {audio_path}: {e}")


def main():
    setup_logging()

    base_dir = 'C:/Workspace-DoHyeonLim/PythonWorkspace/Lieon-ai/Dataset/[Temp]total_data'
    audio_dir = os.path.join(base_dir, "Audio")
    output_csv_dir = os.path.join(base_dir, "Label")

    use_auth_token = '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'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    pipeline = load_pipeline(use_auth_token, device)

    audio_files = [
        os.path.join(audio_dir, f)
        for f in os.listdir(audio_dir)
        if f.endswith('.wav')
    ]

    args_list = [(audio_file, output_csv_dir, pipeline) for audio_file in audio_files]

    # 병렬 처리 수행 (프로세스 기반)
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_single_file, args) for args in args_list]
        for future in as_completed(futures):
            pass  # 개별 작업 완료 시 추가 처리가 필요하면 여기에 작성

    logging.info("All files have been processed.")


if __name__ == "__main__":
    main()
