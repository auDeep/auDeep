#!/bin/bash
# Copyright (C) 2017-2019 Michael Freitag, Shahin Amiriparian, Sergey Pugachevskiy, Nicholas Cummins, Björn Schuller
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

taskName="compare19BS"
workspace="audeep_workspace"

# base directory for audio files
audio_base=".."

##########################################################
# 1. Spectrogram Extraction
##########################################################

# We use 80 ms Hann windows to compute spectrograms
window_width="0.08"

# We use 40 ms overlap between windows to compute spectrograms
window_overlap="0.04"

# Mel-scale spectrograms with 128 frequency bands are extracted
mel_bands="128"

# The ComParE 2019 Baby Sounds (BS) audio files differ in length. By setting the --fixed-length option, we make sure that all
# audio files are exactly 1 seconds long. This is achieved by cutting or zero-padding audio files as required.
fixed_length="1"

# We filter low amplitudes in the spectrograms, which eliminates some background noise. Our system normalises
# spectrograms so that the maximum amplitude is 0 dB, and we filter amplitudes below -40 dB, -50 dB, -60 dB and -70 dB.
clip_below_values="-40 -50 -60 -70"

# Parser for the data set
parser="audeep.backend.parsers.compare19_bs.Compare19BSParser"

# Base path for spectrogram files. auDeep automatically creates the required directories for us.
spectrogram_base="${workspace}/input/spectrograms"

for clip_below_value in ${clip_below_values}; do
    spectrogram_file="${spectrogram_base}/${taskName}-${window_width}-${window_overlap}-${mel_bands}${clip_below_value}.nc"

    if [ ! -f ${spectrogram_file} ]; then
        echo audeep preprocess${verbose_option} --parser ${parser} --basedir ${audio_base} --output ${spectrogram_file} --window-width ${window_width} --window-overlap ${window_overlap} --fixed-length ${fixed_length} --center-fixed --clip-below ${clip_below_value} --mel-spectrum ${mel_bands}
        echo
        audeep preprocess${verbose_option} --parser ${parser} --basedir ${audio_base} --output ${spectrogram_file} --window-width ${window_width} --window-overlap ${window_overlap} --fixed-length ${fixed_length} --center-fixed --clip-below ${clip_below_value} --mel-spectrum ${mel_bands}
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
# Train for 16 epochs, in batches of 1024 examples
num_epochs="16"
batch_size="1024"

# Use learning rate 0.001 and keep probability 80% (20% dropout).
learning_rate="0.001"
keep_prob="0.8"

# Base path for training runs. auDeep automatically creates the required directories for us.
output_base="${workspace}/output"

# Train one autoencoder on each type of spectrogram
for clip_below_value in ${clip_below_values}; do
    # The file containing the extracted spectrograms
    spectrogram_file="${spectrogram_base}/${taskName}-${window_width}-${window_overlap}-${mel_bands}${clip_below_value}.nc"

    # Base directory for the training run
    run_name="${output_base}/${taskName}-${window_width}-${window_overlap}-${mel_bands}${clip_below_value}/t-${num_layers}x${num_units}-${bidirectional_encoder_key}-${bidirectional_decoder_key}"

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
    spectrogram_file="${spectrogram_base}/${taskName}-${window_width}-${window_overlap}-${mel_bands}${clip_below_value}.nc"

    # Base directory for the training run
    run_name="${output_base}/${taskName}-${window_width}-${window_overlap}-${mel_bands}${clip_below_value}/t-${num_layers}x${num_units}-${bidirectional_encoder_key}-${bidirectional_decoder_key}"

    # Models are stored in the "logs" subdirectory of the training run base directory
    model_dir="${run_name}/logs"

    # The file to which we write the learned representations
    representation_file="${run_name}/representations.nc"

    if [ ! -f ${representation_file} ]; then
        echo audeep${verbose_option} t-rae generate --model-dir ${model_dir} --input ${spectrogram_file} --output ${representation_file}
        echo
        audeep${verbose_option} t-rae generate --model-dir ${model_dir} --input ${spectrogram_file} --output ${representation_file}
    fi
done

##########################################################
# 4. Feature Fusion
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
# 5. Feature Export
##########################################################
export_base="${workspace}/csv"
if [ ! -f ${export_base} ]; then
	mkdir $export_base
fi
export_basename="ComParE2019_BabySounds.auDeep"

for clip_below_value in ${clip_below_values}; do
    # Base directory for the training run
    run_name="${output_base}/${taskName}-${window_width}-${window_overlap}-${mel_bands}${clip_below_value}/t-${num_layers}x${num_units}-${bidirectional_encoder_key}-${bidirectional_decoder_key}"

    # The file containing the learned representations
    representation_file="${run_name}/representations.nc"

    # The filenames for the CSV feature sets
    export_name="${export_basename}${clip_below_value}"

    echo audeep export --input ${representation_file} --format CSV --labels-last --output "${export_base}/partitions" --name ${export_name}
    echo
    audeep export --input ${representation_file} --format CSV --labels-last --output "${export_base}/partitions" --name ${export_name}

    # Copy features to the main csv directory of the subchallenge
    if [ ! -f "${audio_base}/features" ]; then
		mkdir ${audio_base}/features
	fi
    cp "${export_base}/partitions/train/${export_name}.csv" "${audio_base}/features/${export_name}.train.csv"
    cp "${export_base}/partitions/devel/${export_name}.csv" "${audio_base}/features/${export_name}.devel.csv"
    cp "${export_base}/partitions/test/${export_name}.csv" "${audio_base}/features/${export_name}.test.csv"
done

export_name="${export_basename}-fused"

echo audeep export --input ${fused_file} --format CSV --labels-last --output "${export_base}/partitions" --name ${export_name}
echo
audeep export --input ${fused_file} --format CSV --labels-last --output "${export_base}/partitions" --name ${export_name}

# Copy features to the main csv directory of the subchallenge
cp "${export_base}/partitions/train/${export_name}.csv" "${audio_base}/features/${export_name}.train.csv"
cp "${export_base}/partitions/devel/${export_name}.csv" "${audio_base}/features/${export_name}.devel.csv"
cp "${export_base}/partitions/test/${export_name}.csv" "${audio_base}/features/${export_name}.test.csv"
