#!/bin/bash
# Copyright (C) 2017  Michael Freitag, Shahin Amiriparian, Sergey Pugachevskiy, Nicholas Cummins, Björn Schuller
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with auDeep.  If not, see <http://www.gnu.org/licenses/>.

verbose_option=""

# Uncomment for debugging of auDeep
# verbose_option=" --verbose --debug"

# Uncomment for debugging of shell script
# set -x;

export PYTHONUNBUFFERED=1

taskName="gtzan"

# base directory for audio files
audio_base="${taskName}/input/data_set"

#########################################################
# 0. PREPARATION
#########################################################

# Check if the GTZAN data set is present at the expected location. If not, download it.
gtzan_file="${audio_base}/genres.tar.gz"

# We act as if there is a train partition containing all instances, so that we can use the PartitionedParser later
dummy_train_partition_dir="${audio_base}/train"

if [ ! -d ${audio_base} -o ! -f ${gtzan_file} ]; then
    echo "GTZAN data set tarball not found at ${gtzan_file}"

    mkdir -p ${audio_base}

    echo "Retrieving GTZAN data set from http://opihi.cs.uvic.ca/sound/genres.tar.gz"
    wget -O ${gtzan_file} "http://opihi.cs.uvic.ca/sound/genres.tar.gz"
fi

tar xzfv ${gtzan_file} -C ${audio_base}
mv ${audio_base}/genres ${dummy_train_partition_dir}

# Check if auDeep has been properly installed
audeep --version > /dev/null || (echo "Could not execute 'audeep --version' - please check your installation"; exit 1)

##########################################################
# 1. Spectrogram Extraction
##########################################################

# We use 80 ms Hann windows to compute spectrograms
window_width="0.08"

# We use 40 ms overlap between windows to compute spectrograms
window_overlap="0.04"

# Mel-scale spectrograms with 320 frequency bands are extracted
mel_bands="320"

# The GTZAN audio files differ in length by a small number of samples. By setting the --fixed-length option, we make
# sure that all audio files are exactly 31 seconds long. We intentionally select a fixed length that is longer than
# the actual file length. We will split the samples into 15 chunks of 2 seconds each. If we fix the length to exactly
# 30 seconds, there might be too few samples in the last window if the number of windows is not evenly divided by 15.
# By setting a fixed length of 31 seconds, we make sure that there are excess samples, most of which will be discarded
# during chunking
fixed_length="31"

# We split the instances into 15 chunks, each 2 seconds long. Our preliminary experiments have shown that music genre
# classification works better with shorter chunks.
chunk_count="15"
chunk_length="2"

# We filter low amplitudes in the spectrograms, which eliminates some background noise. During our preliminary
# evaluation, we found that different classes have high accuracy if amplitudes below different thresholds are
# filtered. Our system normalises spectrograms so that the maximum amplitude is 0 dB, and we filter amplitudes below
# -30 dB, -45 dB, -60 dB and -75 dB. Furthermore, we have found that including spectrograms without amplitude clipping
# helps classification accuracy for music genre classification. For the sake of simplicity of this script, we "simulate"
# omitting the --clip-below option by setting it to a very low amplitude value.
clip_below_values="-30 -45 -60 -75 -1000"

# Base path for spectrogram files. auDeep automatically creates the required directories for us.
spectrogram_base="${taskName}/input/spectrograms"

for clip_below_value in ${clip_below_values}; do
    spectrogram_file="${spectrogram_base}/${taskName}-${window_width}-${window_overlap}-${mel_bands}${clip_below_value}-${chunk_count}-${chunk_length}.nc"

    if [ ! -f ${spectrogram_file} ]; then
        echo audeep preprocess${verbose_option} --basedir ${audio_base} --output ${spectrogram_file} --window-width ${window_width} --window-overlap ${window_overlap} --fixed-length ${fixed_length} --chunk-count ${chunk_count} --chunk-length ${chunk_length} --clip-below ${clip_below_value} --mel-spectrum ${mel_bands}
        echo
        audeep preprocess${verbose_option} --basedir ${audio_base} --output ${spectrogram_file} --window-width ${window_width} --window-overlap ${window_overlap} --fixed-length ${fixed_length} --chunk-count ${chunk_count} --chunk-length ${chunk_length} --clip-below ${clip_below_value} --mel-spectrum ${mel_bands}
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

# Use a bidirectional encoder and unidirectional decoder. Since this is set via a command line flags instead of command
# line options, we set a key value here which is translated into the correct flags below.
bidirectional_encoder_key="b"
bidirectional_decoder_key="x"
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
for clip_below_value in ${clip_below_values}; do
    # The file containing the extracted spectrograms
    spectrogram_file="${spectrogram_base}/${taskName}-${window_width}-${window_overlap}-${mel_bands}${clip_below_value}-${chunk_count}-${chunk_length}.nc"

    # Base directory for the training run
    run_name="${output_base}/${taskName}-${window_width}-${window_overlap}-${mel_bands}${clip_below_value}-${chunk_count}-${chunk_length}/t-${num_layers}x${num_units}-${bidirectional_encoder_key}-${bidirectional_decoder_key}"

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
for clip_below_value in ${clip_below_values}; do
    # The file containing the extracted spectrograms
    spectrogram_file="${spectrogram_base}/${taskName}-${window_width}-${window_overlap}-${mel_bands}${clip_below_value}-${chunk_count}-${chunk_length}.nc"

    # Base directory for the training run
    run_name="${output_base}/${taskName}-${window_width}-${window_overlap}-${mel_bands}${clip_below_value}-${chunk_count}-${chunk_length}/t-${num_layers}x${num_units}-${bidirectional_encoder_key}-${bidirectional_decoder_key}"

    # Models are stored in the "logs" subdirectory of the training run base directory
    model_dir="${run_name}/logs"

    # The file to which we write the learned representations
    representation_file="${output_base}/${taskName}-${window_width}-${window_overlap}-${mel_bands}${clip_below_value}-${chunk_count}-${chunk_length}/t-${num_layers}x${num_units}-${bidirectional_encoder_key}-${bidirectional_decoder_key}/representations.nc"
    representation_file_cv="${output_base}/${taskName}-${window_width}-${window_overlap}-${mel_bands}${clip_below_value}-${chunk_count}-${chunk_length}/t-${num_layers}x${num_units}-${bidirectional_encoder_key}-${bidirectional_decoder_key}/representations.cv.nc"

    if [ ! -f ${representation_file} ]; then
        echo audeep${verbose_option} t-rae generate --model-dir ${model_dir} --input ${spectrogram_file} --output ${representation_file}
        echo
        audeep${verbose_option} t-rae generate --model-dir ${model_dir} --input ${spectrogram_file} --output ${representation_file}
    fi

    # We add a randomly generated stratified 5-fold cross-validation setup to the data, since there is no predefined
    # cross-validation setup
    if [ ! -f ${representation_file_cv} ]; then
        echo audeep modify --add-cv-setup 5 --input ${representation_file} --output ${representation_file_cv}
        echo
        audeep modify --add-cv-setup 5 --input ${representation_file} --output ${representation_file_cv}
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
for clip_below_value in ${clip_below_values}; do
    # The file to containing the learned representations
    representation_file_cv="${output_base}/${taskName}-${window_width}-${window_overlap}-${mel_bands}${clip_below_value}-${chunk_count}-${chunk_length}/t-${num_layers}x${num_units}-${bidirectional_encoder_key}-${bidirectional_decoder_key}/representations.cv.nc"

    # The file to which we write classification accuracy
    results_file="${output_base}/${taskName}-${window_width}-${window_overlap}-${mel_bands}${clip_below_value}-${chunk_count}-${chunk_length}/t-${num_layers}x${num_units}-${bidirectional_encoder_key}-${bidirectional_decoder_key}/results.csv"

    echo "ACCURACY,UAR" > ${results_file}

    echo audeep mlp evaluate --quiet --input ${representation_file_cv} --majority-vote --cross-validate --shuffle --num-epochs ${mlp_num_epochs} --learning-rate ${mlp_learning_rate} --keep-prob ${mlp_keep_prob} --num-layers ${mlp_num_layers} --num-units ${mlp_num_units} --repeat ${mlp_repeat} | tee -a ${results_file}
    echo
    audeep mlp evaluate --quiet --input ${representation_file_cv} --majority-vote --cross-validate --shuffle --num-epochs ${mlp_num_epochs} --learning-rate ${mlp_learning_rate} --keep-prob ${mlp_keep_prob} --num-layers ${mlp_num_layers} --num-units ${mlp_num_units} --repeat ${mlp_repeat} | tee -a ${results_file}
done

##########################################################
# 5. Feature Fusion
##########################################################

# File to which we write the fused representations
fused_file="${output_base}/${taskName}-fused/representations.nc"
fused_file_cv="${output_base}/${taskName}-fused/representations.cv.nc"

# Fuse all learned representations
if [ ! -f ${fused_file} ]; then
    echo audeep${verbose_option} fuse --input ${output_base}/${taskName}-${window_width}-${window_overlap}-${mel_bands}*/*/representations.nc --output ${fused_file}
    echo
    audeep${verbose_option} fuse --input ${output_base}/${taskName}-${window_width}-${window_overlap}-${mel_bands}*/*/representations.nc --output ${fused_file}
fi

if [ ! -f ${fused_file_cv} ]; then
    echo audeep modify --add-cv-setup 5 --input ${fused_file} --output ${fused_file_cv}
    echo
    audeep modify --add-cv-setup 5 --input ${fused_file} --output ${fused_file_cv}
fi

##########################################################
# 6. Fused Feature Evaluation
##########################################################

# File to which we write classification results on the fused representations
results_file="${output_base}/${taskName}-fused/results.csv"

echo "ACCURACY,UAR" > ${results_file}
echo audeep mlp evaluate --quiet --input ${fused_file_cv} --majority-vote --cross-validate --shuffle --num-epochs ${mlp_num_epochs} --learning-rate ${mlp_learning_rate} --keep-prob ${mlp_keep_prob} --num-layers ${mlp_num_layers} --num-units ${mlp_num_units} --repeat ${mlp_repeat} | tee -a ${results_file}
echo
audeep mlp evaluate --quiet --input ${fused_file_cv} --majority-vote --cross-validate --shuffle --num-epochs ${mlp_num_epochs} --learning-rate ${mlp_learning_rate} --keep-prob ${mlp_keep_prob} --num-layers ${mlp_num_layers} --num-units ${mlp_num_units} --repeat ${mlp_repeat} | tee -a ${results_file}
