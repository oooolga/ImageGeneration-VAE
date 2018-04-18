#!/usr/bin/env bash
if [[ ${PWD} != */ImageGeneration-VAE ]]
    then
        echo current dir is ${PWD}
        echo Please go to project root directory!
        exit
fi

export PYTHONUNBUFFERED=1
PYTHON_BIN="/u/luyuchen/miniconda2/envs/pytorch3/bin/python" # change to yours
#PYTHON_BIN="/home/luyuchen/.conda/envs/pytorch/bin/python" # change to yours
{PYTHON_BIN} train.py
