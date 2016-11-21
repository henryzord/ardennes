## Ardennes

Ardennes is an algorithm for performing the classification task using datasets with numerical attributes categorical classes. This is currently the only set of configuration possible, but is soon expected to be enhanced for any type of predictive attribute.

It supports both multiclass and binary classifications.

## Usage

### First steps

To perform a 10-fold cross-validation in one of the toy datasets provided along the algorithm, simply run the ```__main__.py``` script under the root folder:

```sh
    python train.py
```

The expected output should be something like this:
```sh
iter: 000	mean: +0.257922	median: +0.251082	max: +0.424242
iter: 001	mean: +0.319567	median: +0.272727	max: +0.424242
iter: 002	mean: +0.331169	median: +0.277056	max: +0.424242
...
```

Where ```iter``` is the current iteration of the algorithm, and ```mean```, ```median``` and ```max``` are the corresponding properties of the fitness distribution for the current iteration.

### Further usage

To use it for a custom dataset, you may find the algorithm under ```treelib.Ardennes```.
