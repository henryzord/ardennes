#!/usr/bin/env bash

# -- essential -- #
pip install networkx liac-arff numpy scikit-learn pandas scipy
# -- for plotting trees and interpreting graphical models -- #
sudo apt-get install graphviz libgraphviz-dev pkg-config
pip install pygraphviz matplotlib plotly
# -- for running j48 inside python -- #
sudo apt-get install default-jre default-jdk
pip install additional_packages/python-weka-wrapper-0.3.9.tar.gz
# -- for parallel processing - greatly increases performance -- #

# -- for cuda -- #
pip install pycuda
# -- for opencl -- #
sudo apt-get install libffi-dev g++
sudo apt-get install ocl-icd-opencl-dev  # TODO may need some changes in future
pip install mako
## then follow instructions from https://wiki.tiker.net/PyOpenCL/Installation/Linux :
## download http://pypi.python.org/pypi/pyopencl
## unpack it :
# tar xfz pyopencl-VERSION.tar.gz
# cd pyopencl-VERSION
# python configure.py
# su -c "make install"
## and you're done!
