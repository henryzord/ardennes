#!/usr/bin/env bash
conda create --name env_ardennes python=3 --yes
source activate env_ardennes
conda install --file requirements.txt --yes
pip install liac-arff pyopencl -q
cd device
python setup.py install
cd ..