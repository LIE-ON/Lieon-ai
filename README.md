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
<strong>1. Collecting</strong>
<br>
<p>We collected <strong>10 hours</strong> of Korean voice phishing data from YouTube. All data has been checked for duplicates and anomalies, and unnecessary sound effects have been removed. Furthermore, to ensure consistent training, all data has been segmented into multiple 10-second clips.</p>
<br>

<strong>2. Labeling</strong>
<br>
<p>For Speaker Diarization, we utilized a pretrained model provided by the <a href="https://github.com/pyannote/pyannote-audio">Pyannote</a> library.</p>
<ul>
 <li>The voices of the scam callers(voice phishing scammers) were labeled as 1, </li>
 <li>And the voices of the recipients(ordinary conversation) were labeled as 0.</li>
</ul>
<br>

<strong>3. Augmentation</strong>
<br>
<p>We tried augmentation method to expand the amount of data.<br><strong>Time strech</strong>, <strong>pitch shift</strong> and <strong>adding noise</strong> were used to augmetation.</p>
<br>
<p>Using these methods, we also utilized <strong>40 hours</strong> of augmented data for training.</p>
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

## Model
<p><strong>Classifier : Echo State Network</strong></p>
<img width="540" src="https://github.com/user-attachments/assets/106042b5-dc88-474a-8013-058f4a150e21">
<ul>
    <li>A specific kind of recurrent neural network (RNN) designed to efficiently handle sequential data based on Reservoir Computing.</li>
    <li>Considering the need for a model with low computational requirements for real-time AI predictions during calls and the ability to reflect the temporal nature of the data, ESN is the most suitable choice.</li>
</ul>
<br>

## Evaulation
<p><strong>Performance Metrics</strong></p>
<img src="https://github.com/user-attachments/assets/122ced5a-7397-4867-b596-643eb39bb07d">
<p>The ESN-based model demonstrates superior performance compared to other machine learning and deep learning models, with significantly faster inference speed than deep learning models. However, due to the limited amount of data, the SVM outperformed the deep learning-based models (this trend is expected to reverse as the data size increases).</p>
<br>
<p><strong>Hyperparameter</strong></p>
<img src="https://github.com/user-attachments/assets/8d39f79a-2ae8-4983-975e-b9f1273eb28f">