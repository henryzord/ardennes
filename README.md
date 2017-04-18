# Ardennes

Ardennes is an Estimation of Distribution Algorithm for performing decision-tree induction, as presented in the paper


>CAGNINI, Henry E. L; BARROS, R. C; BASGALUPP, M. P. Estimation of Distribution Algorithms for Decision-Tree Induction. IEEE Congress on Evolutionary Computation (IEEE CEC 2017), San Sebastián, Spain, June 5-8, 2017. 

## Citation

If you find this code useful in your work, please cite it:

```bibtex
@inproceedings{cagnini2017ardennes,
  author    = {Henry E. L. Cagnini and
               Rodrigo C. Barros and
               M\'{a}rcio P. Basgalupp},
  title     = {{Estimation of Distribution Algorithms for Decision-Tree Induction}},
  booktitle = {{IEEE} Congress on Evolutionary Computation, {CEC} 2017, San Sebastián, Spain, June 5-8, 2017},
  year      = {2017}
}
```

## Capabilities
* Datasets with numerical predictive attributes
* Categorical class attributes
* Multiclass and binary problems
* More types of datasets will be added in next iterations of the algorithm

## Limitations

This algorithm will only work:
* for datasets with class at the last attribute;
* for datasets with numerical predictive attributes, and categorical class attribute;
* only binary splits;
* Tested only on Ubuntu 16.04, but will probably work in any other SO once you figure out the corresponding libraries described in [Installation]().

## Installation

Essential:
```sh
pip install networkx liac-arff numpy scikit-learn pandas scipy
```
For plotting trees and interpreting graphical models:
```sh
sudo apt-get install graphviz libgraphviz-dev pkg-config
pip install pygraphviz matplotlib plotly
```
For running j48 inside python:
```sh
sudo apt-get install default-jre default-jdk
pip install additional_packages/python-weka-wrapper-0.3.9.tar.gz
```
For parallel processing - greatly increases performance:
```sh
sudo apt-get install libffi-dev g++
sudo apt-get install ocl-icd-opencl-dev
pip install mako
```
Then follow instructions from https://wiki.tiker.net/PyOpenCL/Installation/Linux, or optionally:

**NOTICE:** If you use a virtual environment, you must activate it before running the following commands. 

```sh
tar xfz additional_packages/pyopencl-2016.2.1.tar.gz
cd pyopencl-2016.2.1
python configure.py
sudo su -c "make install"
```
And you're done!

## First steps

Your starting point should be by taking a look at the code located at the `main.py` script. Once you figure out what it does (it is fairly simple to understand), you can call it from terminal:

```sh
python main.py
```

The expected output should be something like this:
```sh
NOTICE: Using single-threaded CPU as device.
training ardennes for dataset liver-disorders
iter: 000 mean: 0.690761 median: 0.688406 max: 0.818841 ET: 22.49sec  height:  9  n_nodes: 45  test acc: 0.536232
iter: 001 mean: 0.674928 median: 0.692029 max: 0.818841 ET:  4.09sec  height:  9  n_nodes: 45  test acc: 0.536232
...
iter: 099 mean: 0.730978 median: 0.789855 max: 0.818841 ET: 2.68sec  height:  9  n_nodes: 27  test acc: 0.637681
Test acc: 0.64 Height: 9 n_nodes: 27 Time: 342.39 secs
```

* The first line (_NOTICE: Using single-threaded CPU as device._) denotes which processor you are using to compute the splitting criterion and individual's fitness. Currently there are two possible processors: OpenCL and single-threaded CPU, which is obviously slower.
* The second line brings information about the dataset over which Ardennes is training.
* The rest of the output is explained as follows:
  * **iter:** current iteration/generation
  * **mean:** mean training accuracy in the current population
  * **median:** median training accuracy in the current population
  * **max:** maximum training accuracy in the current population
  * **ET:** estimated time that this generation took to process
  * **height:** height of the best individual in the current population
  * **n_nodes:** number of nodes of the best individual in the current population
  * **test acc:** test accuracy of the best individual in the current population. This information is **not used** during the evolutionary process; it is only displayed for clarity purposes, and is available if you pass a test set to Ardennes.

## Structure of the code

* `config.json`: Where you will input the algorithm parameters, such as **number of individuals, number of iterations, decile and maximum tree height**.
* `main.py`: starting point for running the algorithm.
* `evaluate.py`: the module which is called from `main.py`. It has several functions which perform holdout, cross-validation and such operations.
* `treelib`: directory for the main Ardennes code. 