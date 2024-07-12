# Lieon-ai
Echo State Network(ESN) for Real-Time Lie Detection in Voice Domain
<br><br>
<img width="1070" alt="architecture" src="https://github.com/LIE-ON/Lieon-ai/assets/94499717/df8a1c03-f246-4754-be18-517897ecdb1e">

## Data
<strong>1. Labeling</strong>
<br>
<br>

<strong>2. Augmentation</strong>
<br>
<p>We tried augmentation method to expand the amount of data.<br><strong>Time strech</strong>, <strong>pitch shift</strong> and <strong>adding noise</strong> were used to augmetation.</p>
<br>
<br>

<strong>3. Generation</strong>
<p>To deal with the lack of data despite augmentation, we used generative AI for producing audio data which have biological features similar to the original data.
We conducted a data generation experiment using the two models below:</p>
<a href='https://github.com/LimDoHyeon/AAGAN'>AAGAN</a> : Audio-to-Audio Generative Adversarial Networks (made by <a href='https://github.com/LimDoHyeon'>Do-Hyeon Lim</a>)

<a href='https://github.com/LimDoHyeon/MVGAN'>MVGAN</a> : Audio-to-Audio GAN using Mel-spectrogram Generator and Vocoder (made by <a href='https://github.com/LimDoHyeon'>Do-Hyeon Lim</a>)
<br>
<br>

## Feature
<ul>
 <li>MFCC(total 20 of feature vectors)</li>
 <li>Pitch</li>
 <li>F0(Fundamental Frequency)</li>
 <li>Spectral Flux</li>
 <li>Spectral Frequency</li>
</ul>

## Model
Echo State Network
<br><br>

## Evaulation
<br><br>

## Reference
[1]https://doi.org/10.48550/arXiv.1712.04323 (Github : https://github.com/stefanonardo/pytorch-esn) <br>
[2]https://doi.org/10.48550/arXiv.2010.05646 (Github : https://github.com/jik876/hifi-gan)
