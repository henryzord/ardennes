from termcolor import colored

try:
    # noinspection PyUnresolvedReferences
    import pyopencl
    from opencl import CLDevice as AvailableDevice

    print colored('NOTICE: Using OpenCL as device.', 'yellow')
except ImportError:
    from __base__ import  Device as AvailableDevice
    print colored('NOTICE: Using single-threaded CPU as device.', 'yellow')

# from termcolor import colored
# from __base__ import  Device as AvailableDevice
# print colored('NOTICE: Using single-threaded CPU as device.', 'yellow')
