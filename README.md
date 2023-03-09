# ChemGraphs
![Build](https://img.shields.io/travis/alejogiley/chemgraphs.svg)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/a968993d062d479b8984e06017ebe950)](https://www.codacy.com/gh/alejogiley/ChemGraphs/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=alejogiley/ChemGraphs&amp;utm_campaign=Badge_Grade)
[![Coverage Status](https://coveralls.io/repos/github/alejogiley/ChemGraphs/badge.svg?branch=prototype)](https://coveralls.io/github/alejogiley/ChemGraphs?branch=prototype)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Predicting inhibitory response of ligands using Graph convolutional networks (GCNs) trained on censored data.

## Background

Censored datasets are common in drug discovery and other fields where the data being measured is below a certain detection limit. Bioactivity assays are typically performed over a limited range of compound concentrations therefore some $IC50$ or $EC50$ values may be reported as being above or below a maximum or minimum concentration, resulting in censored data.

When analyzing censored data, traditional regression models may lead to biased estimates and incorrect conclusions. This is because traditional regression models assume that the censored values are missing at random, which is often not the case in practice. Ignoring the censored data or using imputation methods to estimate the missing values can also result in biased estimates and incorrect conclusions.

Censored regression models, on the other hand, are specifically designed to handle censored data. These models take into account the fact that the censored data is not missing at random and use likelihood-based methods to estimate the parameters of the model. This leads to unbiased estimates and more accurate predictions of the binding affinities.

## Censored Datasets

Using a [*Tobit*](https://en.wikipedia.org/wiki/Tobit_model) model as the loss function is a common approach when dealing with censored data in regression problems. The tobit model is a type of censored regression model that takes into account the censored values and estimates the parameters of the model using maximum likelihood methods.

Using a graph neural network (GNN) to predict the binding affinity from the ligand 2D structure is also a promising approach. GNNs are designed to handle graph-structured data, such as molecular structures, and have been shown to be effective in predicting molecular properties and activities.

By combining a GNN with a tobit loss function, this approach can improve the accuracy of the predictions and provide a powerful tool for drug discovery.

## Installation

To install all dependencies, first you need to create a conda enviroment using the environment file provided here:

```bash
conda env create -f environment.yml
conda activate ChemGraph
```

To download and install the latest version from github:

```bash
git clone https://github.com/alejogiley/ChemGraphs.git
cd ChemGraphs
pip install .
```

## Usage

The bioactivity datasets are provided in the `datasets` directory in SDF format. Data is taken from the [*The Binding Database*](https://www.bindingdb.org/bind/index.jsp).

To process the raw dataset into a format suitable for training the predictive model, you can use the `setup_dataset.py` application. You can select the metric type to use for the predictions, e.g. IC50 or Ki.

```bash
setup_dataset.py \
    --binding datasets/estrogen_receptor.sdf \
    --data_path datasets \
    --file_name "estrogen_dataset" \
    --metric_type "IC50"
```

To train a model, you can use the `train_gcnn.py` application. This script takes as input a formated dataset and trains a GCNN model using the tobit loss function. The model is saved to the specified path. You can define the number of epochs, batch size, learning rate, number of channels and layers, and the seed for the random number generator.

```bash
train_gcnn.py \
    --data_path datasets/estrogen_dataset.lz4 \
    --record_path history.csv \
    --model_path model.h5 \
    --metrics_path metrics.dat \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-2 \
    --channels 64 16 \
    --n_layers 2 \
    --seed 0 \
    "maxlike_tobit_loss" 
```

## Testing

To run the *unit* and integration tests, you can use the `test.sh` script provided here:

```bash
bash test.sh
```