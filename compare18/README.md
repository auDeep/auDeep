# 2018 Computational Paralinguistics Challenge (ComParE)

This package contains shell scripts to generate feature sets for the 2018 Computational Paralinguistics Challenge with auDeep. These scripts are meant to be part of the official ComParE 2018 distribution. As such, there is no evaluation built into the scripts. Instead, the evaluation facilities provided with the ComParE 2018 data sets should be used.

## Installation

Install auDeep as per the instructions provided with the auDeep distribtion.

## Methodology

The shell scripts employ the default auDeep pipeline for generating features. That is, spectrograms are first extracted from the raw audio files. Subsequently, a sequence-to-sequence autoencoder is trained on these spectrograms. Finally, the learned representations are extracted from the trained autoencoder and exported as ARFF files. 

The baseline scripts provided with the ComParE 2018 data sets (under the folder `baseline_audeep`) can then be used to obtain evaluation results on these feature sets.

## Generating Features

For convenience, we provide a shell script that performs the entire feature generation process. For this, follow the steps outlined below.

1. Activate the auDeep virtualenv (`source <path-to-virtualenv>/bin/activate`)
2. `cd` into the `baseline_audeep` folder for the respective subchallenge
3. Run the `audeep_generate.sh` script

This will start the feature generation pipeline. Once the pipeline has completed, the generated features will be saved in the folder `./audeep-workspace/arff`. Additionally, the features will be copied to the main `arff` directory of the respective subchallenge. Follow the specific instructions for the respective subchallenge to obtain evaluation results on these features.