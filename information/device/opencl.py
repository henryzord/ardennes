#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pyopencl as cl
import os

from __base__ import Device


class CLDevice(Device):
    MIN_N_THREADS = 32
    MAX_N_THREADS = 1024

    def __init__(self, dataset):
        super(CLDevice, self).__init__(dataset)

        kernel = open(os.path.join(self._split, 'kernel.cl'), 'r').read()

        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.flags = cl.mem_flags
        self.mem_dataset = cl.Buffer(
            self.ctx, self.flags.READ_ONLY | self.flags.COPY_HOST_PTR, hostbuf=dataset.values.ravel()
        )  # transfers dataset to device memory

        self.mem_class_labels = cl.Buffer(
            self.ctx, self.flags.READ_ONLY | self.flags.COPY_HOST_PTR, hostbuf=self.numerical_class_labels
        )  # transfers class labels to dataset

        self.prg = cl.Program(self.ctx, kernel).build()  # builds program; registers functions

        self._func_gain_ratio = self.prg.gain_ratio

    def device_gain_ratio(self, subset_index, attribute, candidates):
        n_candidates = candidates.shape[0]
        candidates = candidates.astype(np.float32)

        n_threads = n_candidates
        if n_candidates > CLDevice.MAX_N_THREADS:
            raise NotImplementedError('Support for higher than %d threads per kernel launch not implemented!' % CLDevice.MAX_N_THREADS)

        _mem_candidates = cl.Buffer(
            self.ctx, self.flags.READ_WRITE | self.flags.COPY_HOST_PTR, hostbuf=candidates
        )
        _mem_subset_index = cl.Buffer(
            self.ctx, self.flags.READ_ONLY | self.flags.COPY_HOST_PTR, hostbuf=subset_index.astype(np.int32)
        )

        global_size = (n_threads, )  # TODO optimize!
        local_size = (CLDevice.MIN_N_THREADS, )  # TODO optimize!

        self._func_gain_ratio(  # returns an event, for blocking
            self.queue,
            global_size,
            local_size,
            self.mem_dataset,
            np.int32(self.n_objects),
            np.int32(self.n_attributes),
            _mem_subset_index,
            np.int32(self.attribute_index[attribute]),
            np.int32(n_candidates),
            _mem_candidates,
            np.int32(self.class_labels.shape[0]),
            self.mem_class_labels
        )

        cl.enqueue_copy(self.queue, candidates, _mem_candidates)  # returns an event, for blocking

        return candidates

    def device_predict_object(self):
        pass

        # __kernel void predict_objects(
        # __global float *dataset, int n_objects, int n_attributes,
        # __global int *attribute_index, __global float *thresholds,
        # int n_individuals, int H, __global int *predictions) {