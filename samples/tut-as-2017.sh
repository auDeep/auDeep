#!/bin/bash
# Copyright (C) 2017-2018 Michael Freitag, Shahin Amiriparian, Sergey Pugachevskiy, Nicholas Cummins, Björn Schuller
#
# This file is part of auDeep.
#
# auDeep is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# auDeep is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with auDeep. If not, see <http://www.gnu.org/licenses/>.

verbose_option=""

# Uncomment for debugging of auDeep
# verbose_option=" --verbose --debug"

# Uncomment for debugging of shell script
# set -x;

export PYTHONUNBUFFERED=1

taskName="tut-as-2017"

# base directory for audio files
audio_base="${taskName}/input/data_set"

#########################################################
# 0. PREPARATION
#########################################################

tut_as_files="TUT-acoustic-scenes-2017-development.audio.1.zip TUT-acoustic-scenes-2017-development.audio.10.zip TUT-acoustic-scenes-2017-development.audio.2.zip TUT-acoustic-scenes-2017-development.audio.3.zip TUT-acoustic-scenes-2017-development.audio.4.zip TUT-acoustic-scenes-2017-development.audio.5.zip TUT-acoustic-scenes-2017-development.audio.6.zip TUT-acoustic-scenes-2017-development.audio.7.zip TUT-acoustic-scenes-2017-development.audio.8.zip TUT-acoustic-scenes-2017-development.audio.9.zip TUT-acoustic-scenes-2017-development.doc.zip TUT-acoustic-scenes-2017-development.error.zip TUT-acoustic-scenes-2017-development.meta.zip"
tut_as_download_baseurl="https://zenodo.org/record/400515/files/"

# Check if the TUT AS 2017 data set is present at the expected location. If not, download it.
for file in ${tut_as_files}; do
    if [ ! -f "${audio_base}/${file}" ]; then
        wget -O "${audio_base}/${file}" "${tut_as_download_baseurl}/${file}"
    fi

    unzip "${audio_base}/${file}" -d "${audio_base}"
done

mv ${audio_base}/TUT-acoustic-scenes-2017-development/* "${audio_base}/"
rm -r "${audio_base}/TUT-acoustic-scenes-2017-development"

# Check if auDeep has been properly installed
audeep --version > /dev/null || (echo "Could not execute 'audeep --version' - please check your installation"; exit 1)

##########################################################
# 1. Spectrogram Extraction
##########################################################

# We use 160 ms Hann windows to compute spectrograms
window_width="0.16"

# We use 80 ms overlap between windows to compute spectrograms
window_overlap="0.08"

# Mel-scale spectrograms with 128 frequency bands are extracted
mel_bands="320"

# Unlike the other data sets, the TUT AS 2017 does not benefit from amplitude clipping. Instead, we have found that
# extracting spectrograms from different combinations of the two channels present in the audio files benefits
# classification accuracy quite substantially.
channels="mean diff left right"

# Base path for spectrogram files. auDeep automatically creates the required directories for us.
spectrogram_base="${taskName}/input/spectrograms"

for channel in ${channels}; do
    spectrogram_file="${spectrogram_base}/${taskName}-${window_width}-${window_overlap}-${mel_bands}-${channel}.nc"

    if [ ! -f ${spectrogram_file} ]; then
        echo audeep preprocess${verbose_option} --basedir ${audio_base} --output ${spectrogram_file} --window-width ${window_width} --window-overlap ${window_overlap} --channels ${channel} --mel-spectrum ${mel_bands}
        echo
        audeep preprocess${verbose_option} --basedir ${audio_base} --output ${spectrogram_file} --window-width ${window_width} --window-overlap ${window_overlap} --channels ${channel} --mel-spectrum ${mel_bands}
    fi
done

##########################################################
# 2. Autoencoder Training
##########################################################

# Network topology:
# =================
# Use two RNN layers in both the encoder and decoder
num_layers="2"

# Use 256 units per RNN layer
num_units="256"

# Use GRU cells
cell="GRU"

# Use a unidirectional encoder and bidirectional decoder. Since this is set via a command line flags instead of command
# line options, we set a key value here which is translated into the correct flags below.
bidirectional_encoder_key="x"
bidirectional_decoder_key="b"
bidirectional_encoder_option=""
bidirectional_decoder_option=""

if [ ${bidirectional_encoder_key} == "b" ]; then
    bidirectional_encoder_option=" --bidirectional-encoder"
fi

if [ ${bidirectional_decoder_key} == "b" ]; then
    bidirectional_decoder_option=" --bidirectional-decoder"
fi

# Network training:
# =================
# Train for 40 epochs, in batches of 512 examples
num_epochs="40"
batch_size="512"

# Use learning rate 0.001 and keep probability 80% (20% dropout).
learning_rate="0.001"
keep_prob="0.8"

# Base path for training runs. auDeep automatically creates the required directories for us.
output_base="${taskName}/output"

# Train one autoencoder on each type of spectrogram
for channel in ${channels}; do
    # The file containing the extracted spectrograms
    spectrogram_file="${spectrogram_base}/${taskName}-${window_width}-${window_overlap}-${mel_bands}-${channel}.nc"

    # Base directory for the training run
    run_name="${output_base}/${taskName}-${window_width}-${window_overlap}-${mel_bands}-${channel}/t-${num_layers}x${num_units}-${bidirectional_encoder_key}-${bidirectional_decoder_key}"

    # Directory for storing temporary files. The spectrograms are temporarily stored as TFRecords files, in order to
    # be able to leverage TensorFlows input queues. This substantially improves training speed at the cost of using
    # additional disk space.
    temp_dir="${run_name}/tmp"

    if [ ! -d ${run_name} ]; then
        echo audeep${verbose_option} t-rae train --input ${spectrogram_file} --run-name ${run_name} --tempdir ${temp_dir} --num-epochs ${num_epochs} --batch-size ${batch_size} --learning-rate ${learning_rate} --keep-prob ${keep_prob} --cell ${cell} --num-layers ${num_layers} --num-units ${num_units}${bidirectional_encoder_option}${bidirectional_decoder_option}
        echo
        audeep${verbose_option} t-rae train --input ${spectrogram_file} --run-name ${run_name} --tempdir ${temp_dir} --num-epochs ${num_epochs} --batch-size ${batch_size} --learning-rate ${learning_rate} --keep-prob ${keep_prob} --cell ${cell} --num-layers ${num_layers} --num-units ${num_units}${bidirectional_encoder_option}${bidirectional_decoder_option}
    fi
done

##########################################################
# 3. Feature Generation
##########################################################

# For each trained autoencoder, extract the learned representations of spectrograms
for channel in ${channels}; do
    # The file containing the extracted spectrograms
    spectrogram_file="${spectrogram_base}/${taskName}-${window_width}-${window_overlap}-${mel_bands}-${channel}.nc"

    # Base directory for the training run
    run_name="${output_base}/${taskName}-${window_width}-${window_overlap}-${mel_bands}-${channel}/t-${num_layers}x${num_units}-${bidirectional_encoder_key}-${bidirectional_decoder_key}"

    # Models are stored in the "logs" subdirectory of the training run base directory
    model_dir="${run_name}/logs"

    # The file to which we write the learned representations
    representation_file="${output_base}/${taskName}-${window_width}-${window_overlap}-${mel_bands}-${channel}/t-${num_layers}x${num_units}-${bidirectional_encoder_key}-${bidirectional_decoder_key}/representations.nc"

    if [ ! -f ${representation_file} ]; then
        echo audeep${verbose_option} t-rae generate --model-dir ${model_dir} --input ${spectrogram_file} --output ${representation_file}
        echo
        audeep${verbose_option} t-rae generate --model-dir ${model_dir} --input ${spectrogram_file} --output ${representation_file}
    fi
done

##########################################################
# 4. Feature Evaluation
##########################################################

# MLP topology:
# =============
# Use two hidden layers with 150 units each
mlp_num_layers="2"
mlp_num_units="150"

# MLP training:
# =============
# Train for 400 epochs without batching
mlp_num_epochs="400"

# Use learning rate 0.001 and keep probability 60% (40% dropout)
mlp_learning_rate="0.001"
mlp_keep_prob="0.6"

# Repeat evaluation five times and report the average accuracy.
mlp_repeat="5"

# For each trained autoencoder, evaluate a MLP classifier on the extracted features
for channel in ${channels}; do
    # The file to containing the learned representations
    representation_file="${output_base}/${taskName}-${window_width}-${window_overlap}-${mel_bands}-${channel}/t-${num_layers}x${num_units}-${bidirectional_encoder_key}-${bidirectional_decoder_key}/representations.nc"

    # The file to which we write classification accuracy
    results_file="${output_base}/${taskName}-${window_width}-${window_overlap}-${mel_bands}-${channel}/t-${num_layers}x${num_units}-${bidirectional_encoder_key}-${bidirectional_decoder_key}/results.csv"

    echo "ACCURACY,UAR" > ${results_file}

    echo audeep mlp evaluate --quiet --input ${representation_file} --cross-validate --shuffle --num-epochs ${mlp_num_epochs} --learning-rate ${mlp_learning_rate} --keep-prob ${mlp_keep_prob} --num-layers ${mlp_num_layers} --num-units ${mlp_num_units} --repeat ${mlp_repeat} | tee -a ${results_file}
    echo
    audeep mlp evaluate --quiet --input ${representation_file} --cross-validate --shuffle --num-epochs ${mlp_num_epochs} --learning-rate ${mlp_learning_rate} --keep-prob ${mlp_keep_prob} --num-layers ${mlp_num_layers} --num-units ${mlp_num_units} --repeat ${mlp_repeat} | tee -a ${results_file}
done

##########################################################
# 5. Feature Fusion
##########################################################

# File to which we write the fused representations
fused_file="${output_base}/${taskName}-fused/representations.nc"

# Fuse all learned representations
if [ ! -f ${fused_file} ]; then
    echo audeep${verbose_option} fuse --input ${output_base}/${taskName}-${window_width}-${window_overlap}-${mel_bands}*/*/representations.nc --output ${fused_file}
    echo
    audeep${verbose_option} fuse --input ${output_base}/${taskName}-${window_width}-${window_overlap}-${mel_bands}*/*/representations.nc --output ${fused_file}
fi

##########################################################
# 6. Fused Feature Evaluation
##########################################################

# File to which we write classification results on the fused representations
results_file="${output_base}/${taskName}-fused/results.csv"

echo "ACCURACY,UAR" > ${results_file}
echo audeep mlp evaluate --quiet --input ${fused_file} --cross-validate --shuffle --num-epochs ${mlp_num_epochs} --learning-rate ${mlp_learning_rate} --keep-prob ${mlp_keep_prob} --num-layers ${mlp_num_layers} --num-units ${mlp_num_units} --repeat ${mlp_repeat} | tee -a ${results_file}
echo
audeep mlp evaluate --quiet --input ${fused_file} --cross-validate --shuffle --num-epochs ${mlp_num_epochs} --learning-rate ${mlp_learning_rate} --keep-prob ${mlp_keep_prob} --num-layers ${mlp_num_layers} --num-units ${mlp_num_units} --repeat ${mlp_repeat} | tee -a ${results_file}
