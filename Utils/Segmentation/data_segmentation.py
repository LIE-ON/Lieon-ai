import os
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor

# 10초 기준의 초 길이
CHUNK_DURATION = 10


def create_output_dir(input_dir):
    output_dir = f"{input_dir}_edit"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def split_audio(file_path, output_dir):
    file_name = os.path.basename(file_path)
    file_stem, ext = os.path.splitext(file_name)

    # 오디오 파일 불러오기
    audio = AudioSegment.from_file(file_path)

    # 오디오 파일 길이 계산 (ms 단위)
    audio_duration = len(audio) / 1000
    num_chunks = int(audio_duration // CHUNK_DURATION)

    # 데이터 자르기
    for i in range(num_chunks):
        chunk = audio[i * CHUNK_DURATION * 1000: (i + 1) * CHUNK_DURATION * 1000]
        chunk_name = f"{file_stem}_{i + 1}{ext}"
        chunk.export(os.path.join(output_dir, chunk_name), format=ext.replace('.', ''))

    # 남은 부분이 있다면 마지막 조각으로 저장
    if audio_duration % CHUNK_DURATION != 0:
        chunk = audio[num_chunks * CHUNK_DURATION * 1000:]
        chunk_name = f"{file_stem}_{num_chunks + 1}{ext}"
        chunk.export(os.path.join(output_dir, chunk_name), format=ext.replace('.', ''))


def process_directory(input_dir):
    output_dir = create_output_dir(input_dir)

    # 파일 리스트 가져오기
    audio_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.wav', '.mp3'))]

    # 병렬 처리를 위한 스레드 풀
    with ThreadPoolExecutor() as executor:
        for file_path in audio_files:
            executor.submit(split_audio, file_path, output_dir)


if __name__ == "__main__":
    input_directory = "actual_directory"
    process_directory(input_directory)
