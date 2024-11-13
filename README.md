# Lieon-ai
Real-time Voice Phishing(Lie) Classifier using Echo State Networks
<br><br>
<img width="100%" alt="architecture" src="https://github.com/user-attachments/assets/37a04bbe-ee7b-4ecd-b0ec-c9eaa7bf8909">

## Requirements
<p>All code was written in <strong>Python>=3.7.</strong></p>
<p>To download the libraries used in this project, enter the following command:</p>
<blockquote>!pip install -r requirement.txt</blockquote>
<br>

## Data
<strong>1. Labeling</strong>
<br>
<p>For Speaker Diarization, we utilized a pretrained model provided by the <a href="https://github.com/pyannote/pyannote-audio">Pyannote</a> library.</p>
<ul>
 <li>The voices of the scam callers(voice phishing scammers) were labeled as 1, </li>
 <li>And the voices of the recipients were labeled as 0.</li>
</ul>
<br>

<strong>2. Augmentation</strong>
<br>
<p>We tried augmentation method to expand the amount of data.<br><strong>Time strech</strong>, <strong>pitch shift</strong> and <strong>adding noise</strong> were used to augmetation.</p>
<br>

<del><strong>3. Generation</strong></del>
<p><del>To deal with the lack of data despite augmentation, we used generative AI for producing audio data which have biological features similar to the original data.
We conducted a data generation experiment using the two models below:</del></p>
<del><a href='https://github.com/LimDoHyeon/AAGAN'>AAGAN</a> : Audio-to-Audio Generative Adversarial Networks (made by <a href='https://github.com/LimDoHyeon'>Do-Hyeon Lim</a>)</del>

<del><a href='https://github.com/LimDoHyeon/MVGAN'>MVGAN</a> : Audio-to-Audio GAN using Mel-spectrogram Generator and HiFiGAN Vocoder (made by <a href='https://github.com/LimDoHyeon'>Do-Hyeon Lim</a>)</del>
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
<br>

## Model (ongoing)
<p><strong>Classifier : Echo State Network</strong></p>
<img width="540" src="https://github.com/user-attachments/assets/106042b5-dc88-474a-8013-058f4a150e21">
<ul>
 <li>A specific kind of recurrent neural network (RNN) designed to efficiently handle sequential data based on Reservoir Computing.</li>
</ul>
<br>

## Evaulation (ongoing)
<strong>TBA (optimizing)</strong>
<ul>
 <li>Accuracy</li>
 <li>F1 Score</li>
</ul>
<br><br>

## Reference
[1]https://doi.org/10.48550/arXiv.1712.04323 (Github : https://github.com/stefanonardo/pytorch-esn) <br>
[2]https://doi.org/10.48550/arXiv.2010.05646 (Github : https://github.com/jik876/hifi-gan)
