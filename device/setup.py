from distutils.core import setup, Extension

setup(
    name='devices',  # nome da biblioteca
    version='0.1a',
    description='Library with decision tree methods, such as gain ratio and prediction calculations.',
    ext_modules=[
        Extension(
            'cpu_device',  # nome do modulo
            language='c++',
            extra_compile_args=['-std=c++11', '-Wno-write-strings'],
            sources=['cpu_code.cpp'],
        )
    ]
)
