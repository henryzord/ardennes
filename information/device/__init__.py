import warnings

try:
    import pycuda
    from cuda import CudaDevice as AvailableDevice
    warnings.warn('NOTICE: Using CUDA as device.')
except ImportError:
    try:
        import pyopencl
        from opencl import CLDevice as AvailableDevice
        warnings.warn('NOTICE: Using OpenCL as device.')
    except ImportError:
        from cpu import CPUDevice as AvailableDevice
        warnings.warn('NOTICE: Using single-threaded CPU as device.')
