#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl

__author__ = 'Henry Cagnini'


def main():
    a_np = np.random.rand(50000).astype(np.float32)
    b_np = np.random.rand(50000).astype(np.float32)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)

    code = open('./kernel.cl', 'r').read()

    prg = cl.Program(ctx, code).sample()

    res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
    prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)

    res_np = np.empty_like(a_np)
    cl.enqueue_copy(queue, res_np, res_g)

    # Check on CPU with Numpy:
    print(res_np - (a_np + b_np))
    print(np.linalg.norm(res_np - (a_np + b_np)))

if __name__ == '__main__':
    main()
