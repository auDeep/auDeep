[![PyPI version](https://badge.fury.io/py/auDeep.svg)](https://badge.fury.io/py/auDeep)
![PyPI - License](https://img.shields.io/pypi/l/auDeep)

**auDeep** is a Python toolkit for unsupervised feature learning with deep neural networks (DNNs). Currently, the main focus of this project is feature extraction from audio data with deep recurrent autoencoders. However, the core feature learning algorithms are not limited to audio data. Furthermore, we plan on implementing additional DNN-based feature learning approaches.

**(c) 2019-2021 Shahin Amiriparian, Michael Freitag, Maurice Gerczuk, Sergey Pugachevskiy, Björn Schuller: Universität Augsburg**

**(c) 2017-2018 Michael Freitag, Shahin Amiriparian, Maurice Gerczuk, Sergey Pugachevskiy, Nicholas Cummins, Björn Schuller: Universität Passau**
Published under GPLv3, see the LICENSE.md file for details.

Please direct any questions or requests to Shahin Amiriparian (shahin.amiriparian at tum.de) or Michael Freitag (freitagm at fim.uni-passau.de).

# Citing
If you use auDeep or any code from auDeep in your research work, you are kindly asked to acknowledge the use of auDeep in your publications.

> S. Amiriparian, M. Freitag, N. Cummins, and B. Schuller. Sequence to sequence autoencoders for unsupervised representation learning from audio, Proceedings of the Detection and Classification of Acoustic Scenes and Events 2017 Workshop, pp. 17-21, 2017

> M. Freitag, S. Amiriparian, S. Pugachevskiy, N. Cummins, and B. Schuller, “auDeep: Unsupervised Learning of Representations from Audio with Deep Recurrent Neural Networks,” Journal of Machine Learning Research, vol. 18, no. 173, pp. 1–5, 2018

# Installation
You can install auDeep directly via pip:
```bash
pip install audeep
```

This project also ships with a `setup.py` file which allows for installation and automatic dependency resolution through `pip`. We strongly recommend installing the project in its own Python `virtualenv` (see [below](#installing-in-a-virtualenv)). 

## Requirements
The minimal requirements to install auDeep are listed below.

- Python 3.7
- TkInter (`python3-tk` package on Ubuntu, selectable during Python install on Windows) if the interactive Matplotlib backend is used. Check if TkInter is installed and configured correctly with `python3 -c "import _tkinter; import tkinter; tkinter._test()"`. A small window with two buttons should appear. 
- virtualenv (`pip3 install virtualenv`)

### Additional Requirements for GPU Support
The `setup.py` script automatically checks if a compatible CUDA version is available on the system. If GPU support is desired, make sure that the following dependencies are installed **before** installing auDeep.

- CUDA Toolkit 10.0
- cuDNN 7.6.5

By default, the CUDA libraries are required to be available on the system path (which should be the case after a standard installation). If, for some reason, the `setup.py` script fails to detect the correct CUDA version, consider manually installing `tensorflow-gpu` 1.15.2 prior to installing auDeep.

### Python Dependencies (installed automatically during setup)
These Python packages are installed automatically during setup by `pip`, and are just listed for completeness.

- cliff
- liac-arff
- matplotlib
- netCDF4
- pandas
- scipy
- sklearn
- tensorflow 1.15.2
- xarray

## Installing in a virtualenv
The recommended type of installation is through `pip` in a separate virtualenv. This guide outlines the standard installation procedure, assuming the following initial directory structure.
```
.                                      Working directory
└─── auDeep                            Project root directory
     ├─── audeep                       Python source directory
     |    └─── ...
     ├─── samples                      
     |    └─── ...                     Scripts to reproduce experiments reported in our JMLR submission
     ├─── patches                      
     |    └─── fix_import_bug.patch    Patch to fix TensorFlow bug
     ├─── .gitignore
     ├─── LICENSE.md                   License information
     ├─── README.md                    This readme
     └─── setup.py                     Setup script
```
Start by creating a Python virtualenv for the installation
```
> virtualenv -p python3 audeep_virtualenv
```
This will create a folder named `audeep_virtualenv` in the current working directory, containing a minimal Python environment. The name and the location of the virtualenv can be chosen arbitrarily, but, for the purpose of this guide, we assume that it is named `audeep_virtualenv` and stored alongside the project root directory. Subsequently, activate the virtualenv with
```
Linux:
> source audeep_virtualenv/bin/activate

Windows:
> .\audeep_virtualenv\Scripts\activate.bat
```
If everything went well, you should see `(audeep_virtualenv)` prepended to the command prompt, indicating that the virtualenv is active. Continue by installing auDeep with
```
> pip3 install ./auDeep
```
where `auDeep` is the name of the project root directory containing the `setup.py` file. This will fetch all required depencendies and install them in the virtualenv.

This completes installation, and the toolkit CLI can be accessed through the `audeep` binary.
```
> audeep --version
```
The virtualenv can be deactivated at any time using the `deactivate` command.
```
> deactivate
```

# Getting Started
In this section, we provide a step-by-step tutorial on how to perform representation learning with auDeep. A complete documentation of the auDeep command line interface can be found in the [next section](#command-line-interface).

We assume in this guide that auDeep has been installed by following the instructions [above](#installing-in-a-virtualenv), and that the `audeep` executable is accessible from the command line. This can be checked, for instance, by typing `audeep --version`, which should print some copyright information and the application version. 

## Overview
Representation learning with auDeep is performed in several distinct stages.
1. Extraction of spectrograms and data set metadata from raw audio files (`audeep preprocess`)
2. Training of a DNN on the extracted spectrograms (`audeep ... train`)
3. Feature generation using a trained DNN (`audeep ... generate`)
4. Evaluation of generated features (`audeep ... evaluate`)
5. Exporting generated features to CSV/ARFF (`audeep export`)

## Obtaining a Data Set
We use the [ESC-10 data set](https://github.com/karoldvl/ESC-10) for environmental sound classification in this guide, which contains 400 instances from 10 classes. In the command line, navigate to a directory of your choice. In the following, we will assume that any commands are executed from the directory you choose in this step. Retrieve the ESC-10 data set from Github with
```
> git clone https://github.com/karoldvl/ESC-10.git
> pushd ESC-10
> git checkout 553c8f1743b9dba6b282e1323c3ca8fa76923448
> popd
```
This will store the data set in a subfolder called `ESC-10`. As the original ESC-10 repository has been merged with the ESC-50 repository, we have to manually checkout the correct commit.

## Extracting Spectrograms
First of all, we need to extract spectrograms and some metadata from the raw audio files we downloaded during the previous step. In order to get a general overview of the audio files contained in a data set, we can use the following command.
```
> audeep inspect raw --basedir ESC-10
```
This will print some logging messages, and a table containing information about the data set.
```
+-----------------------------------+
| data set information              |
+------------------------+----------+
| number of audio files  |      400 |
| number of labels       |       10 |
| cross validation folds |        5 |
| minimum sample length  |   3.64 s |
| maximum sample length  |   7.24 s |
| sample rate            | 44100 Hz |
| channels               |        1 |
+------------------------+----------+
```
As we can see, the ESC-10 data set contains audio files that are between 3.64 seconds and 7.24 seconds long, contain one channel, and are sampled at 44.1 kHz. 

Next, we are going to determine suitable parameters for spectrogram extraction. Our results have shown that, in general, auDeep requires slightly larger FFT windows during spectrogram extraction than one would usually use to extract, for example, MFCCs. Furthermore, auDeep works well on Mel-spectrograms with a relatively large number of frequency bands. As a reasonable starting point for the ESC-10 data set, we would recommend using 80 ms wide FFT windows with overlap 40 ms, and 128 mel frequency bands. 

In our personal opinion, visual feedback is a great aid in selecting parameters for spectrogram extraction. Use the following command to quickly plot a spectrogram with the parameters recommended above.
```
> audeep preprocess --basedir ESC-10 --window-width 0.08 --window-overlap 0.04 --mel-spectrum 128 --fixed-length 5 --pretend 10
```
Here, the `--window-width 0.08` and `--window-overlap 0.04` options specify the FFT window width and overlap in seconds, respectively. With the `--mel-spectrum 128`, we indicate that 128 mel frequency bands should be extracted, and the `--fixed-length 5` option indicates that we want to extract spectrograms from 5 seconds of audio. If samples are shorter than 5 seconds, they are padded with silence, and if they are longer, they are cut to length. Finally, the `--pretend 10` option tells `audeep` to extract and plot a single spectrogram from the 10th instance in the data set. The ordering of instances is somewhat arbitrary, but deterministic between successive calls to `audeep`.

The command will open a window with a plot of the spectrogram, and an amplitude histogram showing the distribution of amplitudes on the dB scale (auDeep normalizes spectrograms to 0 dB). As you can see from these plots, the audio files in the ESC-10 data set contain quite a bit of background noise. Since this can confuse the representation learning algorithms, we recommend filtering some of this background noise, by clipping amplitudes below a certain threshold. We recommend a threshold around -45 dB to -60 dB as a starting point, which can be specified using the `--clip-below` option.
```
> audeep preprocess --basedir ESC-10 --window-width 0.08 --window-overlap 0.04 --mel-spectrum 128 --fixed-length 5 --clip-below -60 --pretend 10
```
Once you are content with your parameter choices, initiate spectrogram extraction for the full data set by replacing the `--pretend` option with the `--output` option.
```
> audeep preprocess --basedir ESC-10 --window-width 0.08 --window-overlap 0.04 --mel-spectrum 128 --fixed-length 5 --clip-below -60 --output spectrograms/esc-10-0.08-0.04-128-60.nc
```
After the command finishes, the extracted spectrograms have been stored in [netCDF 4](https://www.unidata.ucar.edu/software/netcdf/) format in a file called `spectrograms/esc-10-0.08-0.04-128-60.nc`. Furthermore, since auDeep recognizes the ESC-10 data set, instance labels and the predefined cross-validation setup are stored alongside the spectrograms.

## Autoencoder Training
Next, we are going to train a recurrent sequence to sequence autoencoder on the spectrograms extracted in the previous step. There are numerous parameters that can be used to fine-tune autoencoder training. For the purposes of this guide, we use parameter choices that we have found to work well during our preliminary experiments. However, we encourage the user to experiment with different parameter values. 

We are going to train an autoencoder with 2 recurrent layers (`--num-layers 2`) containing 256 GRU cells (`--num-units 256`, GRU is used by default) in the encoder and decoder. The encoder RNN is going to be unidirectional (default setting), and the decoder RNN is going to be bidirectional (`--bidirectional-decoder`). Training is going to be performed for 64 epochs (`--num-epochs 64`) with learning rate 0.001 (`--learning-rate 0.001`) and 20% dropout (`--keep-prob 0.8`). The batch size during training has to be chosen depending on the amount of memory available, but 64 should be a good starting point (`--batch-size 64`).
```
> audeep t-rae train --input spectrograms/esc-10-0.08-0.04-128-60.nc --run-name output/esc-10-0.08-0.04-128-60/t-2x256-x-b --num-epochs 64 --batch-size 64 --learning-rate 0.001 --keep-prob 0.8 --num-layers 2 --num-units 256 --bidirectional-decoder
```
The `--input` option specifies the spectrogram file which contains training data, which in our case is the spectrogram file generated during the previous step. The `--run-name` option specifies a directory for the training run in which models and logging information are stored. 

If desired, training progress can be tracked in a web browser using TensorBoard, the visualization tool shipped with TensorFlow. Open a new console in the same working directory you used up to now, and execute
```
> tensorboard --reload_interval 2 --logdir output/esc-10-0.08-0.04-128-60/t-2x256-x-b/logs
```
Once TensorBoard has started, navigate to [localhost:6006](http://localhost:6006) in your web browser.

## Feature Generation
After training has finished, the trained autoencoder can be used to generate features from spectrograms. Assuming that you used the file and directory names suggested above, execute the following command.
```
> audeep t-rae generate --model-dir output/esc-10-0.08-0.04-128-60/t-2x256-x-b/logs --input spectrograms/esc-10-0.08-0.04-128-60.nc --output output/esc-10-0.08-0.04-128-60/representations.nc
```
The `--model-dir` option specifies the directory containing TensorFlow checkpoints for the trained autoencoder, which usually is the `logs` subdirectory of the directory passed to the `--run-name` option during autoencoder training. The `--input` option specifies the spectrogram file for which we wish to generate features, and the `--output` option specifies a file in which to store the generated featuers.

The command will extract the learned hidden representation of each spectrogram as its feature vector, and store these features in the output file. Additionally, as the instance labels and cross-validation setup of the ESC-10 data set have been saved in the spectrogram file, they will be stored together with the generated features as well.

## Feature Evaluation
Since instance labels and a cross-validation setup have been stored, we can now use them to evaluate a simple classifier on the learned representations. We are going to use the built-in multilayer perceptron (MLP) for classification, with 2 hidden layers (`--num-layers 2`) and 150 hidden units per layer (`--num-units 150`). Training is going to be performed for 400 epochs (`--num-epochs 400`) with learning rate 0.001 (`--learning-rate 0.001`) and 40% dropout (`--keep-prob 0.6`). No batching is used during MLP training.
```
audeep mlp evaluate --input output/esc-10-0.08-0.04-128-60/representations.nc --cross-validate --shuffle --num-epochs 400 --learning-rate 0.001 --keep-prob 0.4 --num-layers 2 --num-units 150
```
The `--input` option points to a file containing generated features, and the `--cross-validate` option tells the command to perform cross-validated evaluation using the setup stored in that file. The `--shuffle` option specifies that the training data should be shuffled between training epochs, which can improve generalization of the network.

The command will print classification accuracy on each cross validation fold, as well as average classification accuracy and a confusion matrix. If you followed the parameter choices suggested in this guide exactly, accuracy will be around 80% with variations due to random effects.

## Feature Export
Optionally, the learned representations can be exported to CSV or ARFF for further processing, such as classification with an alternate algorithm.
```
audeep export --input output/esc-10-0.08-0.04-128-60/representations.nc --output output/esc-10-0.08-0.04-128-60/csv --format csv
```
The `--input` option points to a file containing generated features, and the `--output` option points to a directory which should contain the exported features. The `--format` option specifies that the features should be exported as CSV files. The command will create one directory for each cross-validation fold below the output directory, and store the features of the instances in a fold below the respective fold directory.

# Command Line Interface
This project exposes a single command line interface, with several subcommands covering different use cases. Running the `audeep` executable without any command line arguments enters interactive mode, in which an arbitrary number of subcommands can be executed. Alternatively, subcommands and their arguments can be passed to the `audeep` executable as command line arguments. To see an overview of available options and commands, use `audeep --help` or `audeep help <command>`.

## Common Options for All Commands
The following options can be used with all commands listed below. 

| Option&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
| ------ | ------- | ----------- |
| `-v, --verbose` | - | Increased verbosity of output, i.e., print all log messages with the `DEBUG` level or higher. By default, log messages with the `INFO` level or higher are printed. |
| `-q, --quiet` | - | Increased verbosity of output, i.e., print all log messages with the `WARNING` level or higher. By default, log messages with the `INFO` level or higher are printed. |
| `--log-file LOG_FILE` | - | Log command output to the specified file. Disabled by default. |
| `-h, --help` | - | Show help message of the command and exit. |
| `--version` | - | Show version of the application and exit. |

## Spectrogram Extraction Commands
The following command performs spectrogram extraction from raw audio files (step 1 [above](#overview)).

### `audeep preprocess`
Extracts metadata, such as labels, cross-validation information, or partition information, from a data set on disk, and extracts spectrograms from raw audio files. The extracted spectrograms together with their metadata are stored in [netCDF 4](https://www.unidata.ucar.edu/software/netcdf/) format. For a description of our data model, see [Data Model](#data-model).

| Option&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
| ------ | ------- | ----------- |
| `--basedir BASEDIR` | required | The data set base directory |
| `--parser PARSER` | `audeep.backend.parsers.MetaParser` | A parser class for the data set file structure. Parsers are used to detect audio files in a data set, and associate metadata such as labels with the audio files. For an overview of available parsers, and how to implement your own parser, see [Parsers](#parsers). For an overview of our data model, see [Data Model](#data-model). |
| `--output FILE` | `./spectrograms.nc` | The output filename. Extracted spectrograms together with their metadata are written in netCDF 4 format to `FILE`. No output is generated when the `--pretend` option is used. |
| `--window-width WIDTH` | `0.04` | The width in seconds of the FFT windows used to extract spectrograms. |
| `--window-overlap WIDTH` | `0.02` | The overlap in seconds between FFT windows used to extract spectrograms. |
| `--mel-spectrum FILTERS` | - | Extract log-scale mel spectrograms using the specified number of filters. By default, log-scale power spectrograms are extracted. |
| `--chunk-length LENGTH` | - | Split audio files into chunks of the specified length in seconds. Must be used together with the `--chunk-count` option. The `audeep fuse chunks` command can be used to combine chunks into a single instance, both before and after feature learning. |
| `--clip-above dB` | - | Clip amplitudes above the specified decibel value. Spectrograms are normalized in such a way that the highest amplitude is 0 dB. |
| `--clip-below dB` | - | Clip amplitudes below the specified decibel value. Spectrograms are normalized in such a way that the highest amplitude is 0 dB. |
| `--chunk-count COUNT` | - | Split audio files into the specified number of chunks. Must be used together with the `--chunk-length` option. If there are more chunks than specified, the excess chunks are discarded. If there are fewer chunks than specified, an error is raised. |
| `--channels CHANNELS` | `MEAN` | How to handle stereo audio files. Valid options are `MEAN`, which extracts spectrograms from the mean of the two channels, `LEFT`, and `RIGHT`, which extract spectrograms from the left and right channel, respectively, and `DIFF`, which extracts spectrograms from the difference of the two channels. |
| `--fixed-length LENGTH` | - | Ensure that all samples have exactly the specified length in seconds, by cutting or padding audio appropriately. For samples that are longer than the specified length, only the first `LENGTH` seconds of audio are used. For samples that are shorter than the specified length, silence is appended at the end. The `--center-fixed` option can be used to modify this behavior. If chunking is used, it can be useful to set the fixed length slightly longer than the product of chunk length and chunk count, to avoid errors due to inconsistent chunk lengths.
| `--center-fixed` | - | Only takes effect when `--fixed-length` is set. By default, samples are cut from the end, or padded with silence at the end. If this option is set, samples are cut equally at the beginning and end, or padded with silence equally at the beginning and end. |
| `--pretend INDEX` | - | Do not process the entire data set. Instead, extract and plot a single spectrogram from the audio file at the specified index (as determined by the data set parser). |

## Training Commands
The following commands are used to train a feature learning DNN on spectrograms (step 2 [above](#overview)).

### Common Options for All Training Commands
The following options apply to all `audeep ... train` commands. Training progress can be monitored using Tensorboard. Assuming a training run with name `NAME` (see the `--run-name` option) has been started in the current working directory, Tensorboard can be started as follows.
```
> tensorboard --reload_interval 2 --logdir NAME/logs
```
The `--reload_interval` option for Tensorboard is not required, but results in more frequent updates of the displayed data.

| Option&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
| ------ | ------- | ----------- |
| `--batch-size BATCH_SIZE` | `64` | The minibatch size for training |
| `--num-epochs NUM_EPOCHS` | `10` | The number of training epochs |
| `--learning-rate RATE` | `1e-3` | The learning rate for the Adam optimiser |
| `--run-name NAME` | `test-run` | A directory for the training run. After each epoch, the autoencoder model is written to the `logs` subdirectory of this folder. The model consists of several files, which all start with `model-<STEP>`, where `STEP` is the training iteration, i.e. `epoch * number of batches`. Furthermore, diagnostic information about the training run is written to the `logs` subdirectory, which can be viewed using TensorBoard. |
| `--input INPUT_FILES...` | required | One or more netCDF 4 files containing spectrograms and metadata, structured according to our data model. If more than one data file is specified, all spectrograms must have the same shape. Each data file is converted to TFRecords format and written to a temporary directory internally (see the `--tempdir` option). |
| `--tempdir DIR` | System temp directory | Directory where temporary files can be written. If specified, this directory must not exist, i.e. it is created by the application, and is deleted after training. Disk space requirements are roughly the same as the input data files. |
| `--continue` | - | If set, training is continued from the latest checkpoint stored under the directory specified by the `--run-name` option. If no checkpoint is found, the command will fail. If not set, a new training run is started, and the command will fail if there are previous checkpoints. |

### `audeep t-rae train`
Train a time-recurrent autoencoder on spectrograms, or other two-dimensional data. A time-recurrent autoencoder processes spectrograms sequentially along the time-axis, processing the instantaneous frequency vectors at each time step. The time-recurrent autoencoder learns to represent entire spectrograms of possibly varying length by a hidden representation vector with fixed dimensionality.

| Option&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
| ------ | ------- | ----------- |
| `--num-layers LAYERS` | `1` | Number of recurrent layers in the encoder and decoder |
| `--num-units UNITS` | `16` | Number of RNN cells in each recurrent layer |
| `--bidirectional-encoder` | - | Use a bidirectional encoder |
| `--bidirectional-decoder` | - | Use a bidirectional decoder |
| `--keep-prob P` | `0.8` | Keep neural network activations with the specified probability, i.e. apply dropout with probability `1-P` |
| `--encoder-noise P` | `0.0` | Corrupt encoder inputs with probability `P`. If `P` is greater than zero, each time step is set to zero with probability `P` |
| `--feed-previous-prob P` | `0.0` | By default, the expected output of the decoder at the previous time step is fed as the input at the current time step. If `P` is greater than zero, the actual decoder output is fed instead of the expected output with probability `P`. |
| `--cell` | `GRU` | The type of RNN cell. Valid choices are `GRU` and `LSTM` |

### `audeep f-rae train`
Train a frequency-recurrent autoencoder on spectrograms, or other two-dimensional data. A frequency-recurrent autoencoder processes spectrograms sequentially along the frequency-axis. It does not process multiple steps along the time-axis simultaneously, i.e. each instantaneous frequency vector is treated as an individual example. The frequency-recurrent autoencoder learns to represent the instantaneous frequency vectors by a hidden representation vector. Thus, it can be used as an additional preprocessing step before training a time-recurrent autoencoder.

| Option&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
| ------ | ------- | ----------- |
| `--freq-window-width WIDTH` | `32` | Split the input frequency vectors into windows with width `WIDTH` and overlap specified by the `--freq-window-overlap` option |
| `--freq-window-overlap OVERLAP` | `24` | Split the input frequency vectors into windows with width specified by the `--freq-window-width` option and overlap `OVERLAP` |
| `--num-layers LAYERS` | `1` | Number of recurrent layers in the encoder and decoder |
| `--num-units UNITS` | `16` | Number of RNN cells in each recurrent layer |
| `--bidirectional-encoder` | - | Use a bidirectional encoder |
| `--bidirectional-decoder` | - | Use a bidirectional decoder |
| `--keep-prob P` | `0.8` | Keep neural network activations with the specified probability, i.e. apply dropout with probability `1-P` |
| `--encoder-noise P` | `0.0` | Corrupt encoder inputs with probability `P`. If `P` is greater than zero, each time step is set to zero with probability `P` |
| `--feed-previous-prob P` | `0.0` | By default, the expected output of the decoder at the previous time step is fed as the input at the current time step. If `P` is greater than zero, the actual decoder output is fed instead of the expected output with probability `P`. |
| `--cell` | `GRU` | The type of RNN cell. Valid choices are `GRU` and `LSTM` |

### `audeep ft-rae train`
Train a frequency-time-recurrent autoencoder on spectrograms, or other two-dimensional data. A frequency-time-recurrent autoencoder first passes spectrograms through a frequency-recurrent encoder. The hidden representation of the frequency-recurrent encoder is then used as the input of a time-recurrent encoder. The hidden representation of the time-recurrent encoder is used as the initial state of a time-recurrent decoder. Finally, the output of the time-recurrent decoder at each time step is used as the input of a frequency-recurrent decoder, which reconstructs the original input spectrogram.

The main difference to training a frequency-recurrent autoencoder with the `audeep f-rae train` command followed by training a time-recurrent autoencoder on the learned features with the `audeep t-rae train` command is that in the `audeep ft-rae train` command, a joint loss function is used.

| Option&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
| ------ | ------- | ----------- |
| `--freq-window-width WIDTH` | `32` | Split the input frequency vectors into windows with width `WIDTH` and overlap specified by the `--freq-window-overlap` option |
| `--freq-window-overlap OVERLAP` | `24` | Split the input frequency vectors into windows with width specified by the `--freq-window-width` option and overlap `OVERLAP` |
| `--num-f-layers LAYERS` | `1` | Number of recurrent layers in the frequency-recurrent encoder and decoder |
| `--num-f-units UNITS` | `64` | Number of RNN cells in each recurrent layer of the frequency-recurrent RNNs |
| `--num-t-layers LAYERS` | `2` | Number of recurrent layers in the time-recurrent encoder and decoder |
| `--num-t-units UNITS` | `128` | Number of RNN cells in each recurrent layer of the time-recurrent RNNs |
| `--bidirectional-f-encoder` | - | Use a bidirectional frequency encoder |
| `--bidirectional-f-decoder` | - | Use a bidirectional frequency decoder |
| `--bidirectional-t-encoder` | - | Use a bidirectional time encoder |
| `--bidirectional-t-decoder` | - | Use a bidirectional time decoder |
| `--keep-prob P` | `0.8` | Keep neural network activations with the specified probability, i.e. apply dropout with probability `1-P` |
| `--cell` | `GRU` | The type of RNN cell. Valid choices are `GRU` and `LSTM` |
| `--f-encoder-noise P` | `0.0` | Corrupt frequency encoder inputs with probability `P`. If `P` is greater than zero, each time step is set to zero with probability `P` |
| `--t-encoder-noise P` | `0.0` | Corrupt time encoder inputs with probability `P`. If `P` is greater than zero, each time step is set to zero with probability `P` |
| `--f-feed-previous-prob P` | `0.0` | By default, the expected output of the frequency decoder at the previous time step is fed as the input at the current time step. If `P` is greater than zero, the actual frequency decoder output is fed instead of the expected output with probability `P`. |
| `--t-feed-previous-prob P` | `0.0` | By default, the expected output of the time decoder at the previous time step is fed as the input at the current time step. If `P` is greater than zero, the actual time decoder output is fed instead of the expected output with probability `P`. |

## Generation Commands
The following commands are used to generate features using a trained DNN (step 3 [above](#overview)).

### Common Options for All Generation Commands
The following options apply to all feature generation commands. A trained recurrent autoencoder is used to generate features from spectrograms, or other two-dimensional data. 

| Option&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
| ------ | ------- | ----------- |
| `--batch-size BATCH_SIZE` | `500` | The minibatch size. Typically, the minibatch size can be much larger for feature generation than for training. |
| `--model-dir MODEL_DIR` | required | The directory containing trained models, i.e. the `logs` subdirectory of a training run. |
| `--steps STEPS...` | latest training step | Use models at the specified training steps, and generate one set of features with each model. By default, one set of features using the most recent model is generated. |
| `--input INPUT_FILE` | required | A netCDF 4 files containing spectrograms and metadata, structured according to our data model. | 
| `--output OUTPUTS...` | required | The output file names. One file name is required for each training step specified using the `--steps` option. |

### `audeep t-rae generate`
Generate features using a trained time-recurrent autoencoder. The hidden representation, as learned by the autoencoder, for each spectogram is extracted as a one-dimensional feature vector for the respective instance. The resulting features for each instance as well as instance metadata are once again stored in netCDF 4 format.

### `audeep f-rae generate`
Generate features using a trained frequency-recurrent autoencoder. The hidden representation, as learned by the autoencoder, for each instantaneous frequency vector is extracted as a one-dimensional feature vector for the respective spectrogram time step. As opposed to the other feature generation commands, this command generates two-dimensional output for each instance, i.e. it preserves the time-axis. Since much more data is generated per instance, the default minibatch size is set to `64` for this command. The generated features for each instance as well as instance metadata are once again stored in netCDF 4 format.

### `audeep ft-rae generate`
Generate features using a trained frequency-time-recurrent autoencoder. The hidden representation, as learned by the autoencoder, for each spectogram is extracted as a one-dimensional feature vector for the respective instance. The resulting features for each instance as well as instance metadata are once again stored in netCDF 4 format.

## Evaluation and Prediction Commands
While generated features can easily be exported into CSV/ARFF format for external processing (see [below](#auxiliary-commands)), the application provides the option to directly evaluate a set of generated features using a linear SVM or an MLP. We support evaluation using cross-validation based on predetermined folds, or evaluation using predetermined training, development, and test partitions. Prior to training, instances are shuffled and features are standardised using coefficients computed on the training data. Furthermore, predictions can be generated on unlabelled data and saved in CSV format.

### Common Options for SVM and MLP Evaluation
The following options apply to both the `audeep svm evaluate`, and the `audeep mlp evaluate` commands. Cross-validated evaluation and partitioned evaluation are mutually exclusive, i.e. the `--cross-validate` option must not be set if the `--train-partitions` and `--eval-partitions` options are set.

| Option&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
| ------ | ------- | ----------- |
| `--input INPUT_FILE` | required | A netCDF 4 file containing one-dimensional feature vectors and associated metadata, structured according to our data model. |
| `--cross-validate` | - | Perform cross-validated evaluation. Requires the input data to contain cross-validation information. |
| `--train-partitions PARTITIONS...` | - | Perform partitioned evaluation, and train models on the specified partitions. Requires the input data to contain partition information, and requires the `--eval-partitions` option to be set. Valid partition identifiers are `TRAIN`, `DEVEL`, and `TEST`. |
| `--eval-partitions PARTITIONS...` | - | Perform partitioned evaluation, and train models on the specified partitions. Requires the input data to contain partition information, and requires the `--train-partitions` option to be set. Valid partition identifiers are `TRAIN`, `DEVEL`, and `TEST`. |
| `--repeat N` | `1` | Repeat evaluation `N` times, and report mean results. |
| `--upsample` | - | Upsample instances in the training partitions or splits, so that training occurs with balanced classes. |
| `--majority-vote` | - | Use majority voting over individual chunks to determine the predictions for audio files. If each audio file has only one chunk, this option has no effect. |

### Common Options for SVM and MLP Prediction
The following options apply to both the `audeep svm predict`, and the `audeep mlp predict` commands. 

| Option&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
| ------ | ------- | ----------- |
| `--train-input TRAIN_FILE` | required | A netCDF 4 file containing one-dimensional feature vectors and associated metadata to use as training data, structured according to our data model. Must be fully labelled. |
| `--train-partitions PARTITIONS...` | - | Use only the specified partitions from the training data. If this option is set, the data set may contain unlabelled instances, but the specified partitions must be fully labelled. |
| `--eval-input EVAL_FILE` | required | A netCDF 4 file containing one-dimensional feature vectors and associated metadata for which to generate predictions, structured according to our data model. Even if label information is present in this data set, it is not used. |
| `--eval-partitions PARTITIONS...` | - | Use only the specified partitions from the evaluation data. |
| `--upsample` | - | Upsample instances in the training data, so that training occurs with balanced classes. |
| `--majority-vote` | - | Use majority voting over individual chunks to determine the predictions for audio files. If each audio file has only one chunk, this option has no effect. |
| `--output FILE` | required | Print predictions in CSV to the specified file. Each line of this file will contain the filename followed by a tab character, followed by the nominal predicted label for the filename. |

### `audeep svm evaluate/predict`
Evaluate a set of generated features, or predict labels on some data, using a linear SVM. Internally, this command uses the [LibLINEAR](https://www.csie.ntu.edu.tw/~cjlin/liblinear/) backend of [sklearn](http://scikit-learn.org/) for training SVMs. In addition to the options listed above, this command accepts the following options.

| Option&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
| ------ | ------- | ----------- |
| `--complexity COMPLEXITY` | required | The SVM complexity parameter |

### `audeep mlp evaluate/predict`
Evaluate a set of generated features, or predict labels on some data, using an MLP with softmax output. Currently, the entire data set is copied to GPU memory, and no batching is performed. In addition to the options listed above, this command accepts the following options.

| Option&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
| ------ | ------- | ----------- |
| `--num-epochs NUM_EPOCHS` | `400` | The number of training epochs |
| `--learning-rate RATE` | `1e-3` | The learning rate for the Adam optimiser |
| `--num-layers NUM_LAYERS` | `2` | The number of hidden layers |
| `--num-units NUM_UNITS` | `150` | The number of units per hidden layer |
| `--keep-prob P` | `0.6` |  Keep neural network activations with the specified probability, i.e. apply dropout with probability `1-P` |
| `--shuffle` | - | Shuffle instances between each training epoch |

## Auxiliary Commands
### `audeep export`
Export a data set into CSV or ARFF format. Cross validation and partition information are represented through the output directory structure, whereas filename, chunk number and labels are stored within the CSV or ARFF files (for a description of these attributes, see [Data Model](#data-model)).

| Option&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
| ------ | ------- | ----------- |
| `--input INPUT_FILE` | required | A netCDF 4 file containing instances and associated metadata, structured according to our data model. |
| `--format FORMAT` | required | The output format, one of `CSV` or `ARFF` |
| `--labels-last` | - | By default, metadata is stored in the first four attributes / columns. If this option is set, filename and chunk number will be stored in the first two attributes / columns, and nominal and numeric labels will be stored in the last two attributes / columns. |
| `--name NAME` | - | Name of generated CSV / ARFF files. By default, the name of the input file is used. |
| `--output OUTPUT_DIR` | required | The output base directory. |

If neither partition nor cross validation information is present in the data set, this command simply writes all instances to a  single file with the same name as the input file to the output base directory. Please note that partial cross validation or partition information will be discarded.

If only partition information is present, the command writes the instances of each partition to subdirectories `train`, `devel`, and `test` below the output base directory. For each partition, a single file with the same name as the input file is written.

If only cross validation information is present, the command writes the instances of each fold to subdirectories `fold_<INDEX>` below the output base directory, where `<INDEX>` ranges from `1` to the number of folds. For each cross validation fold, a single file with the same name as the input file is written. Please note that in the case of overlapping folds, some instances will be duplicated.

If both partition and cross validation information is present, the command first creates subdirectories `train`, `devel`, and `test` below the output base directory, and then creates subdirectories `fold_<INDEX>` below the partition directories, where `<INDEX>` ranges from `1` to the number of folds. That is, cross validation folds are assumed to be specified per partition. For each partition and cross validation fold, a single file with the same name as the input file is written to the corresponding directory. Please note that in the case of overlapping folds, some instances will be duplicated.

### `audeep import`
Import a data set from CSV or ARFF files into netCDF 4. A data set without partition or cross-validation information can be imported if the import base directory contains no directories, and a single CSV or ARFF file with the name passed to the command. Partitioned data can be imported if the base directory contains one directory for each partition of the data set, which in turn contain a single CSV or ARFF file with the name passed to the command. Finally, data with cross-validation information can be imported if the base directory contains one directory for each fold, named `fold_N` where `N` indicates the fold number. Each of the fold directories must once again contain a single CSV or ARFF file with the name passed to the command.

CSV or ARFF files may contain metadata columns/attributes specifying filename, chunk number, nominal label, and numeric label of instances. At least a nominal label is required currently. Any columns/attributes that are not recognized as metadata are treated as containing numeric features.

| Option&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
| ------ | ------- | ----------- |
| `--basedir BASEDIR` | required | The base directory from which to import data. |
| `--name NAME` | required | The name of the data set files to import. |
| `--filename-attribute NAME` | `filename` | Name of the filename metadata column/attribute. |
| `--chunk-nr-attribute NAME` | `chunk_nr` | Name of the chunk number metadata column/attribute. |
| `--label-nominal-attribute NAME` | `label_nominal` | Name of the nominal label metadata column/attribute. |
| `--label-numeric-attribute NAME` | `label_numeric` | Name of the numeric label metadata column/attribute. |

### `audeep upsample`
Balance classes in some partitions of a data set. Classes are balanced by repeating instances of classes which are underrepresented in comparison to others. Upsampling is performed in such a way that all classes have approximately the same number of instances within the specified partitions.

| Option&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
| ------ | ------- | ----------- |
| `--input INPUT_FILE` | required | A netCDF 4 file containing instances and associated metadata, structured according to our data model. |
| `--partitions PARTITIONS...` | - | One or more partitions in which classes should be balanced. Any partitions not specified here are left unchanged. If not set, the entire data set is upsampled. |
| `--output OUTPUT_FILE` | required | The output filename. |

### `audeep modify`
Modify data set metadata in various ways.

| Option&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
| ------ | ------- | ----------- |
| `--input INPUT_FILE` | required | A netCDF 4 file containing instances and associated metadata, structured according to our data model. |
| `--output OUTPUT_FILE` | required | The output filename. |
| `--add-cv-setup NUM_FOLDS` | - | Randomly generate a cross-validation setup for the data set with `NUM_FOLDS` evenly-sized non-overlapping folds. |
| `--remove-cv-setup` | - | Remove cross-validation information |
| `--add-partitioning PARTITIONS...` | - | Randomly generate a partitioning setup for the data set with the specified evenly-sized partitions. |
| `--remove-partitioning` | - | Remove partition information |

### `audeep fuse`
Combine several data sets by concatenation along a feature dimension. This command ensures data integrity, by refusing to fuse data sets that have different metadata for one or more instances.

| Option&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
| ------ | ------- | ----------- |
| `--input INPUT_FILES...` | required | Two or more netCDF 4 files containing instances and associated metadata, structured according to our data model. |
| `--dimension DIMENSION` | `generated` | The dimension to concatenate features along. Defaults to `generated`, which is the dimension name used when generating features. In order to view feature dimensions of a data set, the `audeep inspect` command can be used. |
| `--output OUTPUT_FILE` | required | The output filename. |

### `audeep fuse chunks`
Combine audio files which have been split into chunks during spectrogram extraction into single instances, by concatenation along a feature dimension.

| Option&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
| ------ | ------- | ----------- |
| `--input INPUT_FILE` | required | A netCDF 4 files containing instances and associated metadata, structured according to our data model. |
| `--dimension DIMENSION` | `generated` | The dimension to concatenate features along. Defaults to `generated`, which is the dimension name used when generating features. In order to view feature dimensions of a data set, the `audeep inspect` command can be used. |
| `--output OUTPUT_FILE` | required | The output filename. |

### `audeep inspect raw`
Display information about the audio files in a data set that has not yet been converted to netCDF 4 format. This command displays information such as the minimum and maximum length of audio files in the data set, or the different sample rates of audio files in the data set.

| Option&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
| ------ | ------- | ----------- |
| `--basedir BASEDIR` | required | The data set base directory |
| `--parser PARSER` | `audeep.backend.parsers.MetaParser` | A parser class for the data set file structure. Parsers are used to detect audio files in a data set, and associate metadata such as labels with the audio files. For an overview of available parsers, and how to implement your own parser, see [Parsers](#parsers). For an overview of our data model, see [Data Model](#data-model). |

### `audeep inspect netcdf`
Display data set metadata stored in a netCDF 4 file structured according to our data model.

| Option&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
| ------ | ------- | ----------- |
| `--input INPUT_FILE` | required | A netCDF 4 file containing instances and associated metadata, structured according to our data model. |
| `--instance INDEX` | - | Display information about the instance at the specified index in addition to generic data set metadata |
| `--detailed-folds` | -  | Display detailed information about cross-validation folds, if present |

### `audeep validate`
Validate data set integrity constraints. This functionality is provided by a separate command, since integrity constraint checking can be rather time-consuming. For a list of integrity constraints, see [Data Model](#data-model).

| Option&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
| ------ | ------- | ----------- |
| `--input INPUT_FILE` | required | A netCDF 4 file containing instances and associated metadata, structured according to our data model. |
| `--detailed` | - | Display detailed information about instances violating constraints |

# Data Model
Acoustic data sets are often structured in very different ways. For example, some data sets include metadata CSV files, while others may rely on a certain directory structure to specify metadata information, or even a combination of both. In order to be able to provide a unified feature learning and evaluation framework, we have developed a data model for data sets containing instances with arbitrary feature dimensions and metadata associated with individual instances. Furthermore, we provide a set of parsers for common data set structures, as well as the option to implement custom parsers (see [Parsers](#parsers)).

Currently, our data model only supports classification tasks, but we plan on adding support for regression tasks in the near future.

## Data Model Structure
Data sets store instances, which can correspond to either an entire audio file, or a chunk of an audio file. For each instance, the following attributes are stored.

| Attribute (Variable Name) | Value Required | Dimensionality | Description |
| ------------------------- | -------------- | -------------- | ----------- |
| Filename (`FILENAME`) | yes | - | The name of the audio file from which the instance was extracted |
| Chunk Number (`CHUNK_NR`) | yes | - | The index of the chunk which the instance represents. The filename and the chunk number attributes together uniquely identify instances. |
| Nominal Label (`LABEL_NOMINAL`) | no | - | Nominal label of the instance. If specified, the numeric label must be specified as well. |
| Numeric Label (`LABEL_NUMERIC`) | no | - | Numeric label of the instance. If specified, the nominal label must be specified as well. |
| Cross validation folds (`CV_FOLDS`) | yes | number of folds | Specifies cross validation information. For each cross validation fold, this attribute stores whether the instance belongs to the training split (`0`), or the validation split (`1`). We have chosen to represent cross validation information in this way, since we have encountered data sets with overlapping cross validation folds, which can not be represented by simply storing the fold number for each instance. Please note that, while this attribute is required to have a value, this value is allowed to have dimension zero, corresponding to no cross validation information. |
| Partition (`PARTITION`) | no | - | The partition to which the instance belongs (`0`: training, `1`: development, `2`: test) |
| Features (`FEATURES`) | yes | arbitrary | The feature matrix of the instance |

Furthermore, we optionally store a label map, which specifies a mapping of nominal label values to numeric label values. If a label map is given, labels are restricted to values in the map.

Internally, we rely on [xarray](http://xarray.pydata.org) to represent data sets.

## Integrity Constraints
The following integrity constraints must be satisfied by valid data sets.

1. Instances with the same filename must have the same nominal labels
2. Instances with the same filename must have the same numeric labels
3. Instances with the same filename must have the same cross validation information
4. Instances with the same filename must have the same partition information
5. If a label map is given, all nominal labels must be keys in the map, and all numeric labels must be the 
   associated values
6. For each filename, there must be the same number of chunks
7. For each filename, chunk numbers must be exactly [0, ..., num_chunks - 1], i.e. each chunk number must be present
   exactly once.

# Parsers
Parsers read information about an acoustic data set, and generate a list of absolute paths to audio files and their metadata. We provide some parsers for common use cases, and custom parsers can easily be implemented by subclassing `audeep.backend.parsers.base.Parser`.

### `audeep.backend.parsers.no_metadata.NoMetadataParser`
Simply reads all WAV files found below the data set base directory, without parsing any metadata beyond the filename. Please note that the built-in evaluation tools can not be used with data sets generated by this parser.
 
### `audeep.backend.parsers.meta.MetaParser`
A meta-parser which decides intelligently which of the parsers listed below should be used to parse a data set. This parser is the default parser used by the `audeep preprocess` command. Please note that this parser does explicitly *not* include the `NoMetadataParser`, since that parser could parse any of the data sets the other parsers can process. If you want to use the `NoMetadataParser`, specify it explicitly using the `--parser` option of the `audeep preprocess` command.

### `audeep.backend.parsers.dcase.DCASEParser`
Parses the development data set of the [2017 DCASE Acoustic Scene Classification challenge](http://www.cs.tut.fi/sgn/arg/dcase2017/challenge/task-acoustic-scene-classification). This parser requires the following directory structure below the data set base directory.
```
.                                      Data set base directory
├─── audio                             Directory containing audio files
|    └─── ...
├─── evaluation_setup                  Directory containing cross validation metadata
|    ├─── fold1_train.txt              Training partition instances for fold 1
|    ├─── fold2_train.txt              Training partition instances for fold 2
|    ├─── fold3_train.txt              Training partition instances for fold 3
|    ├─── fold4_train.txt              Training partition instances for fold 4
|    └─── ...
├─── meta.txt                          Global metadata file
└─── ...
```

### `audeep.backend.parsers.esc.ESCParser`
Parses the [ESC-10 and ESC-50](https://github.com/karoldvl/paper-2015-esc-dataset) data sets. This parser requires that audio files are sorted into separate directories according to their class. Each class directory name must adhere to the regex `^\d{3} - .+`, i.e. the name must start with three digits, followed by the string `" - "`, followed by the class name. Furthermore, audio file names must start with a digit, which is interpreted as the cross validation fold number.

### `audeep.backend.parsers.urban_sound_8k.UrbanSound8KParser`
Parses the [UrbanSound8K](https://serv.cusp.nyu.edu/projects/urbansounddataset/urbansound8k.html) data set. This parser requires the following directory structure below the data set base directory.
```
.                                      Data set base directory
├─── audio                             Directory containing audio files
|    └─── ...                      
├─── metadata                          Directory containing metadata
|    └─── UrbanSound8K.csv             Metadata file
└─── ...
```

### `audeep.backend.parsers.partitioned.PartitionedParser`
A general purpose parser for data sets which have training, development, and/or test partitions. This parser requires that there are one or more partition directories with names `train`, `devel`, or `test` below the data set base directory, containing another layer of directories indicating the labels of the audio files. The latter directories may only contain WAV files. A simple example of a valid directory structure would be as follows.
 ```
.                                      Data set base directory
├─── train                             Directory containing audio files from the training partition
|    ├─── classA                       Directory containing WAV files with label 'classA'
|    |    └─── ...
|    └─── classB                       Directory containing WAV files with label 'classB'
|         └─── ...
└─── devel                             Directory containing audio files from the development partition
      ├─── classA                       Directory containing WAV files with label 'classA'
      |    └─── ...
      └─── classB                       Directory containing WAV files with label 'classB'
           └─── ...
```

### `audeep.backend.parsers.cross_validated.CrossValidatedParser`
A general purpose parser for data sets which have cross validation information. This parser requires that there are two or more directories with names `fold_N`, where `N` indicates the fold number, below the data set base directory. Each of these must contain another layer of directories indicating the labels of the audio files. The latter directories may only contain WAV files. A simple example of a valid directory structure would be as follows.
 ```
.                                      Data set base directory
├─── fold_1                            Directory containing audio files from fold 1
|    ├─── classA                       Directory containing WAV files with label 'classA'
|    |    └─── ...
|    └─── classB                       Directory containing WAV files with label 'classB'
|         └─── ...
└─── fold_2                            Directory containing audio files from fold 2
      ├─── classA                       Directory containing WAV files with label 'classA'
      |    └─── ...
      └─── classB                       Directory containing WAV files with label 'classB'
           └─── ...
```
