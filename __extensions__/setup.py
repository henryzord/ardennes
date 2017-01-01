from distutils.core import setup, Extension
setup(
    name='make_predictions',  # function name
    version='1.0',
    ext_modules=[Extension('c_individual', ['c_individual.c'])]  # module, file
)