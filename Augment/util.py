import matplotlib.pyplot as plt
import numpy as np
import librosa


def plot_features(original, augmented, sr):
    # Extract MFCCs for comparison
    mfcc_original = librosa.feature.mfcc(y=original, sr=sr, n_mfcc=13)
    mfcc_augmented = librosa.feature.mfcc(y=augmented, sr=sr, n_mfcc=13)

    # Plotting MFCCs using matplotlib
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(mfcc_original, aspect='auto', origin='lower', interpolation='none')
    plt.title('Original MFCC')
    plt.xlabel('Time')
    plt.ylabel('MFCC Coefficients')

    plt.subplot(1, 2, 2)
    plt.imshow(mfcc_augmented, aspect='auto', origin='lower', interpolation='none')
    plt.title('Augmented MFCC')
    plt.xlabel('Time')
    plt.ylabel('MFCC Coefficients')

    plt.tight_layout()
    plt.show()