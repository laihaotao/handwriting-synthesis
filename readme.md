# Handwriting Prediction and Synthesis

This repo implements the handwriting prediction and synthesis networks
described in the paper
"[Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850)"
published by Alex Graves.

## Environment

The codebase has been tested to run on a Mac (10.14.6) or a
Ubutun machine (18.06) successfully. It is recommended to run
on a machine with GPU and cuda installed since it can boost
the training speed.

In order to create a same env, you can use the command:

```
conda env create -f <environment-name>.yml
```

### Mac

The tested Mac uses the following main packages (a detail env
description can be found via the file `env_mac.yml` but it may
contain some unneccessary packages for this repo):

```
python=3.6.10
pytorch=1.4.0=py3.6_0
```

### Ubuntu

The tested Ubuntu comes with a Nivida 1070ti GPU using the following main
packages (a detail env description can be found via the file `env_ubuntu.yml`
but it may contain some unneccessary packages for this repo):

```
python=3.6.10
pytorch=1.4.0=py3.6_0
```

## Run Instruction

Before training you will need to run the `prepare_data.py` file to create
the needed data files by the following command:

```
python prepare_data.py
```

To train the model, run `train.py` file. It supports the following
arguments to control the training process. By default, it is set to
train the prediction network.

```
python train.py -h

usage: train.py [-h] [--task TASK] [--num_epochs NUM_EPOCHS]
                [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE]
                [--timesteps TIMESTEPS] [--hidden_size HIDDEN_SIZE]
                [--mix_components MIX_COMPONENTS] [--K K]
                [--model_dir MODEL_DIR]
optional arguments:
  -h, --help            show this help message and exit
  --task TASK           "prediction" or "synthesis"
  --num_epochs NUM_EPOCHS
                        number of training epochs
  --batch_size BATCH_SIZE
                        batch size
  --learning_rate LEARNING_RATE
                        learning rate for training
  --timesteps TIMESTEPS
                        step in time direction for LSTM
  --hidden_size HIDDEN_SIZE
                        number of hidden size for a LSTM cell
  --mix_components MIX_COMPONENTS
                        number of mixture distribution
  --K K                 number of gaussian functions for attention
  --model_dir MODEL_DIR
                        location to save the result
```

## Result

Result can be visualized via the `results.ipynb` notebook, or can be generated
by using the `generate.py` file. It support the following arguments.
By default, it is set to generate the result from the prediction network.

```
python generate.py -h

usage: generate.py [-h] [--task TASK] [--text TEXT]
                   [--pmodel PMODEL] [--smodel SMODEL]
optional arguments:
  -h, --help       show this help message and exit
  --task TASK      "prediction" or "synthesis"
  --text TEXT      text used for synthesis
  --pmodel PMODEL  path to the trained prediction model
  --smodel SMODEL  path to the trained synthesis model
```
