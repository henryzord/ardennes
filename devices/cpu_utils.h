#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"


char *get_dtype_name(int dtype) {
    int dtypes[] = {
            NPY_BOOL, NPY_INT8, NPY_INT16, NPY_INT32, NPY_LONG, NPY_INT64, NPY_UINT8, NPY_UINT16, NPY_UINT32,
            NPY_ULONG, NPY_UINT64, NPY_FLOAT16, NPY_FLOAT32, NPY_FLOAT64, NPY_LONGDOUBLE, NPY_COMPLEX64, NPY_COMPLEX128,
            NPY_CLONGDOUBLE, NPY_DATETIME, NPY_TIMEDELTA, NPY_STRING, NPY_UNICODE, NPY_OBJECT, NPY_VOID
    };

    char *dtype_names[] = {
            "NPY_BOOL", "NPY_INT8", "NPY_INT16", "NPY_INT32", "NPY_LONG", "NPY_INT64", "NPY_UINT8", "NPY_UINT16", "NPY_UINT32",
            "NPY_ULONG", "NPY_UINT64", "NPY_FLOAT16", "NPY_FLOAT32", "NPY_FLOAT64", "NPY_LONGDOUBLE", "NPY_COMPLEX64", "NPY_COMPLEX128",
            "NPY_CLONGDOUBLE", "NPY_DATETIME", "NPY_TIMEDELTA", "NPY_STRING", "NPY_UNICODE", "NPY_OBJECT", "NPY_VOID"
    };

    for(int i = 0; i < 24; i++) {
        if(dtype == dtypes[i]) {
            return dtype_names[i];
        }
    }
    return NULL;
}

char *get_dtype_name(PyArrayObject *array) {
    // all 24 dtypes of numpy
    int dtypes[] = {
            NPY_BOOL, NPY_INT8, NPY_INT16, NPY_INT32, NPY_LONG, NPY_INT64, NPY_UINT8, NPY_UINT16, NPY_UINT32,
            NPY_ULONG, NPY_UINT64, NPY_FLOAT16, NPY_FLOAT32, NPY_FLOAT64, NPY_LONGDOUBLE, NPY_COMPLEX64, NPY_COMPLEX128,
            NPY_CLONGDOUBLE, NPY_DATETIME, NPY_TIMEDELTA, NPY_STRING, NPY_UNICODE, NPY_OBJECT, NPY_VOID
    };

    char *dtype_names[] = {
            "NPY_BOOL", "NPY_INT8", "NPY_INT16", "NPY_INT32", "NPY_LONG", "NPY_INT64", "NPY_UINT8", "NPY_UINT16", "NPY_UINT32",
            "NPY_ULONG", "NPY_UINT64", "NPY_FLOAT16", "NPY_FLOAT32", "NPY_FLOAT64", "NPY_LONGDOUBLE", "NPY_COMPLEX64", "NPY_COMPLEX128",
            "NPY_CLONGDOUBLE", "NPY_DATETIME", "NPY_TIMEDELTA", "NPY_STRING", "NPY_UNICODE", "NPY_OBJECT", "NPY_VOID"
    };

    int a_dtype = PyArray_DESCR(array)->type_num;

    for(int i = 0; i < 24; i++) {
        if(a_dtype == dtypes[i]) {
            return dtype_names[i];
        }
    }
    return NULL;
}

/**
 * Performs checks for numpy.ndarray objects.
 * @param obj_names
 * @param obj_dtypes
 * @param objs
 * @return
 */
bool perform_checks(char *obj_names[], int obj_dtypes[], PyObject *objs[]) {
    char buffer[200];

    for (int i = 0; obj_names[i] != NULL; i++) {
        if (!PyArray_Check((PyObject *) objs[i])) {
            sprintf(buffer, "%s must be a numpy.ndarray", obj_names[i]);
            PyErr_SetString(PyExc_TypeError, buffer);
            return false;
        }
        if (!PyArray_IS_C_CONTIGUOUS((PyArrayObject*)objs[i]) && !PyArray_IS_F_CONTIGUOUS((PyArrayObject*)objs[i])) {
            sprintf(buffer, "%s must be either a Fortran or C contiguous numpy.ndarray.", obj_names[i]);
            PyErr_SetString(PyExc_TypeError, buffer);
            return false;
        }
        if (obj_dtypes[i] != PyArray_TYPE((PyArrayObject*)objs[i])) {
            sprintf(buffer, "%s values must be %s, currently is %s",
                    obj_names[i], get_dtype_name(obj_dtypes[i]), get_dtype_name((PyArrayObject*)objs[i])
            );
            PyErr_SetString(PyExc_TypeError, buffer);
            return false;
        }
    }
    return true;
}

float select(float false_return, float true_return, char condition) {
    if(condition) {
        return true_return;
    }
    return false_return;
}

PyObject *at(PyArrayObject *table, int x, int y) {
    npy_intp itemsize = PyArray_ITEMSIZE(table);
    npy_intp *dims = PyArray_DIMS(table);
    int ndims = PyArray_NDIM(table);

    char *data = PyArray_BYTES(table);

    int c_contiguous = PyArray_IS_C_CONTIGUOUS(table);
    data += c_contiguous * ((ndims > 1) * (itemsize * ((dims[1] * x) + y)) + (ndims == 1) * (itemsize * x)) +
            (!c_contiguous) * ((ndims > 1) * (itemsize * ((dims[0] * y) + x)) + (ndims == 1) * (itemsize * x));

    return PyArray_GETITEM(table, data);
}

PyObject *at(PyArrayObject *table, int x) {
    return at(table, x, 0);
}