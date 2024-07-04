import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment

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

def time_warp(data, time_warping_factor=1.0):
    """
    오디오 파일의 특정 구간을 늘리거나 줄여서 다양한 발음 속도를 시뮬레이션합니다.
    volume_factor가 0.0보다 작으면 볼륨이 줄어들고, 1.0보다 크면 볼륨이 커집니다. 예를 들어, volume_factor=1.5는 볼륨을 50% 높입니다.
    :param data: 변형할 오디오 데이터 배열.
    :param sampling_rate: 오디오의 샘플링 레이트.
    :param time_warping_factor: 시간 왜곡 비율 (default: 1.0).
    :return: 시간 왜곡된 오디오 데이터.
    """
    time_steps = np.arange(len(data))
    warped_time_steps = np.interp(time_steps,
                                  np.linspace(0, len(data), int(len(data) * time_warping_factor)),
                                  time_steps)
    return np.interp(warped_time_steps, time_steps, data)

def change_volume(data, volume_factor=1.0):
    """
    오디오 파일의 전체 볼륨을 조절합니다. 이는 다양한 음량 환경에서 모델이 잘 작동하도록 만듭니다.
    :param data: 변형할 오디오 데이터 배열.
    :param volume_factor: 볼륨 변화 비율 (default: 1.0).
    :return: 볼륨이 조절된 오디오 데이터.
    """
    audio_segment = AudioSegment(
        data.tobytes(),
        frame_rate=sampling_rate,
        sample_width=data.dtype.itemsize,
        channels=1
    )
    return np.array(audio_segment + volume_factor)


"""
# Example of loading an audio file
file_path = 'path_to_audio_file.wav'
data, sr = librosa.load(file_path)

# Apply augmentations
augmented_data_time_stretch = time_stretch(data, rate=1.1)
augmented_data_pitch_shift = pitch_shift(data, sr, n_steps=2)
augmented_data_noise = add_noise(data, noise_factor=0.01)
augmented_data_time_warp = time_warp(data, sr, time_warping_factor=1.1)
augmented_data_volume = change_volume(data, volume_factor=5.0)

# Save augmented data
sf.write('augmented_time_stretch.wav', augmented_data_time_stretch, sr)
sf.write('augmented_pitch_shift.wav', augmented_data_pitch_shift, sr)
sf.write('augmented_noise.wav', augmented_data_noise, sr)
sf.write('augmented_time_warp.wav', augmented_data_time_warp, sr)
sf.write('augmented_volume.wav', augmented_data_volume, sr)
"""