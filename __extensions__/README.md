### Creating your own extensions

If you want to modify the behavior of the single-thread, CPU-based predictor for decision trees, modify the file ```c_individual.c``` and build it using the following code:

```sh
    python setup.py build
```

To simply install the modified module to your Python, run

```sh
    python setup.py install
```

Notice that if you use a virtual environment (virtual_env or conda), the files may not be written correctly. 


For more information, please refer to the [official Python documentation.](https://docs.python.org/2/extending/building.html)