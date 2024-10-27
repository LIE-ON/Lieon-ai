import os


def compare_audio_label(audio_dir, label_dir):
    # Audio_edit와 Label_edit 디렉토리에서 파일명 가져오기
    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.csv')])

    # 오디오 파일에서 'data'라는 글자를 'label'로 변경하여 배열 생성
    converted_audio_files = [f.replace('data', 'label').replace('.wav', '.csv') for f in audio_files]

    # 라벨 파일 배열과 오디오에서 변환된 파일 배열 비교하여 라벨에만 있는 파일 찾기
    unmatched_labels = [label for label in label_files if label not in converted_audio_files]

    # 결과 출력
    if unmatched_labels:
        print("List of label files that do not correspond to audio files:")
        for label in unmatched_labels:
            print(label)
    else:
        print("All label files correspond 1:1 to audio files.")


if __name__ == "__main__":
    audio_edit_directory = "your_audio_edit_directory"
    label_edit_directory = "your_label_edit_directory"

    compare_audio_label(audio_edit_directory, label_edit_directory)
