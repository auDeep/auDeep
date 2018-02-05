This folder contains scripts which can be used to reproduce the results we report in our JMLR submission 
> M. Freitag, S. Amiriparian, S. Pugachevskiy, N. Cummins, and B.Schuller. auDeep: Unsupervised Learning of Representations from Audio with Deep Recurrent Neural Networks, Journal of Machine Learning Research, 2017, submitted, 5 pages

Furthermore, the scripts serve as an in-depth example of using the auDeep command line interface, which covers more functionality than the getting started guide in the main `README` file. In this document, we provide brief instructions on how to use the scripts, and an overview over our parameter choices. 

# Requirements
All scripts are implemented using the `bash` scripting language, and a `bash` interpreter with access to the GNU coreutils (`mv`, `cp`, etc.) is required to run them. Furthermore, the following programs are used by the scripts and required for them to function correctly.
- Git through the `git` executable
- GNU Wget through the `wget` executable
- GNU tar through the `tar` executable
- Unzip through the `unzip` executable

Furthermore, the auDeep application has to be installed, and accessible through the `audeep` executable (usually by activating the `virtualenv` in which auDeep has been installed). This can be achieved, for example, by following the installation guide in the main `README` file.

# Experiment Scripts
We provide the following experiment scripts.

| Script File | Task |
| ----------- | ---- |
| `esc-10.sh` | Environmental sound classification on the ESC-10 data set |
| `esc-50.sh` | Environmental sound classification on the ESC-50 data set |
| `gtzan.sh` | Music genre classification on the GTZAN data set |
| `tut-as-2017.sh` | Acoustic scene classification on the TUT Acoustic Scenes 2017 data set. Withheld from GitHub until after the DCASE 2017 challenge. |

Each script can be run without prior preparations, and will download the respective data set if required. All files downloaded or created by a script will be stored in a folder with the name of the respective script, e.g. the `esc-10.sh` script will store all its data below the `esc-10` folder in the current working directory. Classification accuracy will be printed both to the command line as well as to `results.csv` files below the `<script name>/output` directory.
 
# Experimental Setup
On all data sets, we pursue a similar approach for our experiments. Several sets of spectrograms are extracted from the raw audio files with different parameters. On each set of spectrograms, we then train a recurrent sequence to sequence autoencoder. Autoencoder parameters are fixed for a given data set, but vary slightly between different data sets. Subsequently, the learned hidden representation of each spectrogram is extracted as its feature vector, resulting in one set of features for each set of spectrograms. For each data set, these feature sets are fused by concatenating the feature vectors of each instance, resulting in the final feature set for each data set. Finally, the built-in multilayer perceptron (MLP) classifier is evaluated on this feature set using cross-validation. We report the average classification accuracy of five repeated cross-validation runs, to reduce the impact of random variations in classification accuracy.

In the following, we give a detailed account of the parameters used for each classification task, as well as describe minor deviations from the scheme described above.

## Common Settings
These settings apply to all tasks, and we will not necessarily document all of them for each individual task.

### Spectrogram Extraction
We extract logarithmic Mel-scale spectrograms, which are normalized to [-1;1] for training. During extraction, spectrograms are normalized to 0 dB. We use symmetric Hann windows for spectrogram extraction, with window widths and overlaps depending on the task. The number of Mel frequency bands that are extracted also depends on the task.

### Autoencoder Training
We use recurrent sequence to sequence autoencoders with two recurrent layers containing 256 GRUs in both the encoder and decoder. Depending on the task, either the encoder or the decoder network are bidirectional. In all cases, we train autoencoders using the Adam optimizer with learning rate 0.001. Furthermore, 20% dropout is applied during training.

### Classification
The MLP configuration used for classification is the same for all tasks. An MLP with 2 hidden layers each containing 150 units with ReLU activation, and a softmax output layer is used. Training is performed for 400 epochs without batching, using the Adam optimizer with learning rate 0.001. During training, 40% dropout is applied, and instances are shuffled between epochs. 

In each cross-validation fold, features in the training split are standardized, and features in the validation split are transformed using the coefficients computed on the training split.

## ESC-10

### Spectrogram Extraction
Instances in the ESC-10 data set vary in length between about 3 to 7 seconds. We always extract spectrograms from 5 seconds of audio, which is achieved by cutting samples that are longer, and by appending silence to samples that are shorter. Cutting and appending is performed in equal parts at the beginning and the end of a sample.

We found that eliminating background noise by clipping amplitudes below a certain threshold in the spectrograms can improve classification accuracy. Since different threshold appear to benefit accuracy on different classes, we extract spectrograms with several thresholds.

| Parameter | Value(s) |
| --------- | -------- |
| FFT window width | 80 ms |
| FFT window overlap | 40 ms |
| Mel frequency bands | 128 |
| clipping thresholds | -30 dB, -45 dB, -60 dB, -75 dB |

### Autoencoder Training
We train autoencoders with unidirectional encoder networks, and bidirectional decoder networks on the extracted spectrograms. Training is performed for 64 epochs in batches of 64 instances.

The autoencoders learn representations of size 1024 (since the state size of the decoder network is 1024), which results in 4096-dimensional feature vectors after fusion of the representations.

### Classification
The ESC-10 data set contains a predefined cross-validation setup, which we use for evaluation.

## ESC-50

### Spectrogram Extraction
Instances in the ESC-50 data set vary in length between about 3 to 7 seconds. We always extract spectrograms from 5 seconds of audio, which is achieved by cutting samples that are longer, and by appending silence to samples that are shorter. Cutting and appending is performed in equal parts at the beginning and the end of a sample.

We found that eliminating background noise by clipping amplitudes below a certain threshold in the spectrograms can improve classification accuracy. Since different threshold appear to benefit accuracy on different classes, we extract spectrograms with several thresholds.

| Parameter | Value(s) |
| --------- | -------- |
| FFT window width | 80 ms |
| FFT window overlap | 40 ms |
| Mel frequency bands | 128 |
| clipping thresholds | -30 dB, -45 dB, -60 dB, -75 dB |

### Autoencoder Training
We train autoencoders with bidirectional encoder networks, and unidirectional decoder networks on the extracted spectrograms. Training is performed for 64 epochs in batches of 64 instances.

The autoencoders learn representations of size 512 (since the state size of the decoder network is 512), which results in 2048-dimensional feature vectors after fusion of the representations.

### Classification
The ESC-50 data set contains a predefined cross-validation setup, which we use for evaluation.

## GTZAN

### Spectrogram Extraction
Instances in the GTZAN data set are about 30 seconds long, which is rather long in comparison to the other data sets. We found that classification accuracy is improved if we split instances into chunks with 2 second length, resulting in 15 chunks per instance. For the purposes of autoencoder training, these chunks are treated as individual instances.

We also discovered that eliminating background noise by clipping amplitudes below a certain threshold in the spectrograms can improve classification accuracy. Since different threshold appear to benefit accuracy on different classes, we extract spectrograms with several thresholds. Furthermore, different from the ESC data sets, we extract spectograms without frequency clipping. In order to reduce the complexity of the experiment script, we actually simulate omitting frequency clipping by clipping amplitudes below -1000 dB.

| Parameter | Value(s) |
| --------- | -------- |
| FFT window width | 80 ms |
| FFT window overlap | 40 ms |
| Mel frequency bands | 320 |
| clipping thresholds | -30 dB, -45 dB, -60 dB, -75 dB, -1000 dB |

### Autoencoder Training
We train autoencoders with bidirectional encoder networks, and unidirectional decoder networks on the extracted spectrograms. Training is performed for 40 epochs in batches of 512 instances (chunks).

The autoencoders learn representations of size 512 (since the state size of the decoder network is 512), which results in 2560-dimensional feature vectors after fusion of the representations.

### Classification
The GTZAN data set does not contain a predefined cross-validation setup. Instead, we generate a stratified 5-fold cross-validation setup, in which all chunk of the same original instance are placed either in the training or the validation split. The MLP classifier is trained to predict labels of the individual chunks, and a majority vote over the chunks of an instance is taken to determine the final classification.

## TUT Acoustic Scenes 2017

### Spectrogram Extraction
As opposed to the other data sets, instances in the TUT Acoustic Scenes 2017 data set are recorded in stereo. We found that in many cases, essential information about the class label is present in only one of the two channels. Therefore, we extract spectrograms from the two channels separately. Furthermore, we extract spectrograms from the mean and difference of the two channels, since we discovered that this further boosts classification accuracy.

Amplitude clipping does not benefit classification accuracy on the TUT Acoustic Scenes 2017 data set much.

| Parameter | Value(s) |
| --------- | -------- |
| FFT window width | 160 ms |
| FFT window overlap | 80 ms |
| Mel frequency bands | 320 |
| channels | left, right, mean, difference |

### Autoencoder Training
We train autoencoders with unidirectional encoder networks, and bidirectional decoder networks on the extracted spectrograms. Training is performed for 40 epochs in batches of 512 instances.

The autoencoders learn representations of size 1024 (since the state size of the decoder network is 1024), which results in 4096-dimensional feature vectors after fusion of the representations.

### Classification
The TUT Acoustic Scenes 2017 data set contains a predefined cross-validation setup, which we use for evaluation.