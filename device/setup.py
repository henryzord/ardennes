from distutils.core import setup, Extension

setup(
    name='devices',  # library name, as it will appear in conda list and pip list commands
    version='0.1a',  # library version, as it will appear in conda list and pip list commands
    description='Library with decision tree methods, such as gain ratio and prediction calculations.',
    ext_modules=[
        Extension(
            'cpu_device',  # module name, as it will appear for import inside python code
            language='c++',
            extra_compile_args=['-std=c++11', '-Wno-write-strings'],
            sources=['cpu_code.cpp'],
        )
    ]
)
