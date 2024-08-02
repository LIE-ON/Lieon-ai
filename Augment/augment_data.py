import os
import librosa
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset, DataLoader


class AudioDataset(Dataset):
    def __init__(self, file_paths, sr=44100):
        self.file_paths = file_paths
        self.sr = sr

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        y, sr = librosa.load(file_path, sr=self.sr)
        return y, sr, os.path.basename(file_path)


def time_stretch(data, rate=1.0):
    """
    오디오 파일의 재생 속도를 변경합니다. 속도가 변화해도 음의 높이는 그대로 유지됩니다.
    rate가 1.0보다 작으면 속도가 느려지고, 1.0보다 크면 속도가 빨라집니다. 예를 들어, rate=1.1은 오디오를 10% 빠르게 재생합니다.
    :param data: 변형할 오디오 데이터 배열.
    :param rate: 시간 변형 비율 (default: 1.0).

    :return: 시간 변형된 오디오 데이터.
    """
    return librosa.effects.time_stretch(data, rate=rate)


def pitch_shift(data, sr, n_steps):
    """
    오디오 파일의 피치를 변경합니다. 재생 속도는 그대로 유지됩니다.
    n_steps가 양수면 피치를 높이고, 음수면 피치를 낮춥니다. 예를 들어, n_steps=2는 피치를 두 반음 올립니다.
    :param data: 변형할 오디오 데이터 배열.
    :param sr: 오디오의 샘플링 레이트.
    :param n_steps: 반음 단위의 피치 변화.
    :return: 피치 변형된 오디오 데이터.
    """
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=n_steps)


def add_noise(data, noise_factor=0.005):
    """
    오디오 파일에 랜덤한 잡음을 추가하여 잡음 환경에서도 모델이 잘 작동하도록 만듭니다.
    :param data: 변형할 오디오 데이터 배열.
    :param noise_factor: 추가할 잡음의 비율 (default: 0.005).
    :return: 잡음이 추가된 오디오 데이터.
    """
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    return augmented_data


def augment_and_save(data, sr, base_filename, output_dir):
    # Augment data 1
    tsf_and_addnoise = time_stretch(data, rate=1.3)
    tsf_and_addnoise = add_noise(tsf_and_addnoise, noise_factor=0.025)
    sf.write(os.path.join(output_dir, f'{base_filename}_augment_1.wav'), tsf_and_addnoise, sr)

    # Augment data 2
    tss_and_addnoise = time_stretch(data, rate=0.7)
    tss_and_addnoise = add_noise(tss_and_addnoise, noise_factor=0.05)
    sf.write(os.path.join(output_dir, f'{base_filename}_augment_2.wav'), tss_and_addnoise, sr)

    # Augment data 3
    psh = pitch_shift(data, sr, n_steps=2)
    sf.write(os.path.join(output_dir, f'{base_filename}_augment_3.wav'), psh, sr)

    # Augment data 4
    psl = pitch_shift(data, sr, n_steps=-2)
    sf.write(os.path.join(output_dir, f'{base_filename}_augment_4.wav'), psl, sr)


def main():
    input_dir = '/Users/imdohyeon/Documents/PythonWorkspace/Lieon-ai/dataset/train'
    output_dir = '/Users/imdohyeon/Documents/PythonWorkspace/Lieon-ai/dataset/Augment'
    os.makedirs(output_dir, exist_ok=True)

    file_paths = [os.path.join(root, file)
                  for root, _, files in os.walk(input_dir)
                  for file in files if file.endswith('.wav')]

    dataset = AudioDataset(file_paths)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    for data, sr, base_filename in dataloader:
        data = data[0].numpy()  # Batch size is 1, so get the first element and convert to numpy array
        sr = sr[0].item()  # Unpack the batch
        base_filename = base_filename[0]  # Unpack the batch
        augment_and_save(data, sr, base_filename, output_dir)

if __name__ == '__main__':
    main()