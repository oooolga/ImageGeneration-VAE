#!/usr/bin/env bash
if [[ ${PWD} != */ImageGeneration-VAE ]]
    then
        echo current dir is ${PWD}
        echo Please go to project root directory!
        exit
fi

export PYTHONUNBUFFERED=1
python train.py