"""
This code splits the entire data(both original and augmented) into train, validation, and test sets.
We suppose that both data are stored in the same directory(data{k} / aug_data{k}).
"""
import os
import shutil
import random


def shuffle_data(audio_files, label_files):
    # 데이터 섞기
    paired_files = list(zip(audio_files, label_files))
    random.shuffle(paired_files)

    # 60%, 20%, 20% 비율로 데이터 나누기
    total_files = len(paired_files)
    train_border = int(total_files * 0.6)
    val_border = int(total_files * 0.8)

    train_files = paired_files[:train_border]
    val_files = paired_files[train_border:val_border]
    test_files = paired_files[val_border:]

    return train_files, val_files, test_files


# 파일을 각각의 디렉토리로 이동하는 함수
def move_files(file_pairs, target_audio_dir, target_label_dir, temp_data_dir):
    for audio_file, label_file in file_pairs:
        # 원본 경로
        src_audio_path = os.path.join(temp_data_dir, 'Audio', audio_file)
        src_label_path = os.path.join(temp_data_dir, 'Label', label_file)

        # 타겟 경로
        dest_audio_path = os.path.join(target_audio_dir, audio_file)
        dest_label_path = os.path.join(target_label_dir, label_file)

        # 파일 이동
        shutil.move(src_audio_path, dest_audio_path)
        shutil.move(src_label_path, dest_label_path)


def main():
    base_dir = '/Users/imdohyeon/Downloads/[Temp]total_data'
    temp_data_dir = os.path.join(base_dir, 'Augmented')
    train_audio_dir = os.path.join(base_dir, 'Train/Audio'); train_label_dir = os.path.join(base_dir, 'Train/Label')
    val_audio_dir = os.path.join(base_dir, 'Val/Audio'); val_label_dir = os.path.join(base_dir, 'Val/Label')
    test_audio_dir = os.path.join(base_dir, 'Test/Audio'); test_label_dir = os.path.join(base_dir, 'Test/Label')

    # [Temp]total_data에서 wav 파일과 그에 대응하는 csv 파일 리스트를 가져옴
    audio_files = sorted([f for f in os.listdir(os.path.join(temp_data_dir, 'Audio')) if f.endswith('.wav')])
    label_files = sorted([f for f in os.listdir(os.path.join(temp_data_dir, 'Label')) if f.endswith('.csv')])

    # 데이터가 정확히 매칭되었는지 확인
    assert len(audio_files) == len(label_files), "The amount of audio and label is different."

    # Shuffle data
    train_files, val_files, test_files = shuffle_data(audio_files, label_files)

    # Execute moving files
    move_files(train_files, train_audio_dir, train_label_dir, temp_data_dir)
    move_files(val_files, val_audio_dir, val_label_dir, temp_data_dir)
    move_files(test_files, test_audio_dir, test_label_dir, temp_data_dir)

    print("Process completed.")


if __name__ == '__main__':
    main()