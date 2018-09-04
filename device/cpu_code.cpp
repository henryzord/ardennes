#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include "numpy/arrayobject.h"

#include <random>

#define true 1
#define false 0

int *update_counters(int n_dims, int *dims, int *counters) {
    char add = 1;
    for(int i = n_dims - 1; i >= 0; i--) {
        if(add) {
            counters[i] += 1;
            add = 0;
        } else {
            break;
        }
        if(counters[i] >= dims[i]) {
            if(i > 0) {
                counters[i] = 0;
                add = 1;
            } else {
                return NULL;
            }
        }
    }
    return counters;
}

const char choice_doc[] = "sample random values from numpy.random.choice";
static PyObject* choice(PyObject *self, PyObject *args, PyObject *kwargs) {

    static char *kwds[] = {"a", "size", "replace", "p", NULL};
    PyObject *size = NULL, *a = NULL, *p = NULL;
    int replace = 1;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OpO", kwds, &a, &size, &replace, &p)) {
        return NULL;
    }

    if(!PyArray_Check(a)) {
        PyErr_SetString(PyExc_TypeError, "a must be a numpy.ndarray");
        return NULL;
    }
    if((p != NULL) && !PyArray_Check(p)) {
        PyErr_SetString(PyExc_TypeError, "p must be a numpy.ndarray");
        return NULL;
    }

    int nd = -1;
    npy_intp *_dims;
    if(size != NULL) {
        if(PyTuple_Check(size)) {
            nd = (int)PyTuple_Size(size);
            _dims = new npy_intp [nd];

            for(int i = 0; i < nd; i++) {
                _dims[i] = (int)PyLong_AsLong(PyTuple_GetItem(size, i));
            }
        } else if (PyLong_Check(size)) {
            nd = 1;
            _dims = new npy_intp [1];
            _dims[0] = (int)PyLong_AsLong(size);
        } else {
            PyErr_SetString(PyExc_TypeError, "size must be either a tuple or an integer");
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_NotImplementedError, "not implemented yet!");
        return NULL;
    }

    int a_ndims = PyArray_NDIM((const PyArrayObject*)a),
        p_ndims = PyArray_NDIM((const PyArrayObject*)p);

    if(a_ndims > 1) {
        PyErr_SetString(PyExc_ValueError, "a must be 1-dimensional");
        return NULL;
    }
    if(p_ndims > 1) {
        PyErr_SetString(PyExc_ValueError, "p must be 1-dimensional");
        return NULL;
    }

    npy_intp *a_dims = PyArray_SHAPE((PyArrayObject*)a),
             *p_dims = PyArray_SHAPE((PyArrayObject*)p);

    for(int j = 0; j < a_ndims; j++) {
        if(a_dims[j] != p_dims[j]) {
            PyErr_SetString(PyExc_ValueError, "a and p must have the same size");
            return NULL;
        }
    }

    PyObject *sampled_obj = PyArray_SimpleNew(nd, _dims, NPY_FLOAT32);
    PyArrayObject *sampled = (PyArrayObject*)sampled_obj;

    if(sampled == NULL) {
        PyErr_SetString(PyExc_ValueError, "Exception ocurred while trying to allocate space for numpy array");
        return NULL;
    }

    npy_intp *sampled_strides = PyArray_STRIDES(sampled),
             *a_strides = PyArray_STRIDES((PyArrayObject*)a),
             *p_strides = PyArray_STRIDES((PyArrayObject*)p);

    int sampled_ndims = PyArray_NDIM(sampled);
    npy_intp *sampled_dims = PyArray_DIMS(sampled);

    char *sampled_ptr, *p_ptr, *a_ptr;  // pointers to data

    int num, div, success = -1;
    float sum, spread = 1000, p_data;
    PyObject *a_data;

//    for(int i = 0; i < sampled_ndims; i++) {
    for(int j = 0; j < sampled_dims[0]; j++) {
        for(int l = 0; l < sampled_dims[1]; l++) {
            sampled_ptr = PyArray_BYTES(sampled);
            sampled_ptr += (sampled_strides[0] * j) + (sampled_strides[1] * l);

            num = rand() % (int)spread;  // random sampled number
            sum = 0;  // sum of probabilities so far

            p_ptr = PyArray_BYTES((PyArrayObject*)p);
            a_ptr = PyArray_BYTES((PyArrayObject*)a);

            success = PyArray_SETITEM(sampled, sampled_ptr, Py_BuildValue("f", -15.0));  // TODO remove!

            for(int k = 0; k < a_dims[0]; k++) {
                break; // TODO remove
                p_data = (float)PyFloat_AsDouble(PyArray_GETITEM((PyArrayObject*)p, p_ptr));
                a_data = PyArray_GETITEM((PyArrayObject*)a, a_ptr);
                p_ptr += p_strides[0];
                a_ptr += a_strides[0];

                div = (int)(num/((sum + p_data) * spread));

                if(div <= 0) {
                    success = PyArray_SETITEM(sampled, sampled_ptr, Py_BuildValue("f", -15.0));
                    printf("success? %d\n", success);
                    break;
                }
                sum += p_data;
            }
        }
    }

    return Py_BuildValue("O", sampled);
}

static int next_node(int current_node, int go_left) {
    return (current_node * 2) + 1 + (!go_left);
}

static void predict_dataset(
    int n_objects, int n_attributes, PyObject *dataset, PyObject *tree, PyObject *predictions,
    PyObject *attribute_index) {

    PyObject *node;
    char *label;

    int int_attr;
    float threshold, value;

    for(int i = 0; i < n_objects; i++) {
        int current_node = 0;

        while(true) {
            node = PyDict_GetItemString(
                PyDict_GetItem(
                    tree, Py_BuildValue("i", current_node)
                ),
                "attr_dict"
            );
            int terminal = PyObject_IsTrue(PyDict_GetItemString(node, "terminal"));

            PyObject *label_object = PyUnicode_AsEncodedString(PyDict_GetItemString(node, "label"), "ascii", "Error ~");
            label =  PyBytes_AsString(label_object);

            printf("terminal? %d label: %s\n", terminal, label);

            if(terminal) {
                PyList_SetItem(predictions, i, label_object);
                break;
            } else {
                threshold = (float)PyFloat_AsDouble(PyDict_GetItemString(node, "threshold"));
                int_attr = (int)PyLong_AsLong(PyDict_GetItemString(attribute_index, label));
                value = (float)PyFloat_AsDouble(PyList_GetItem(dataset, n_attributes + int_attr));

                current_node = next_node(current_node, (value <= threshold));
            }
        }
    }
}

const char make_predictions_doc[] = "Makes predictions for a series of unknown data.\n\n"
    ":param shape: shape of dataset.\n"
    ":param dataset: set of unknown data.\n"
    ":param tree: decision tree which will be used to make the predictions.\n"
    ":param predictions: Empty array in which the predictions will be written.\n"
    ":param attribute_index: A dictionary where the attribute names are the keys and the values are their indexes.\n"
    ":returns: list of predictions, one entry per object passed.";
static PyObject* make_predictions(PyObject *self, PyObject *args) {

    int n_objects, n_attributes;
    PyObject *predictions, *tree, *dataset, *attribute_index, *shape;

    if (!PyArg_ParseTuple(
            args, "O!O!O!O!O!",
            &PyTuple_Type, &shape,
             &PyList_Type, &dataset,
             &PyDict_Type, &tree,
             &PyList_Type, &predictions,
             &PyDict_Type, &attribute_index
    )) {
        return NULL;
    }

    n_objects = (int)PyLong_AsLong(PyTuple_GetItem(shape, 0));
    n_attributes = (int)PyLong_AsLong(PyTuple_GetItem(shape, 1));
//
    predict_dataset(n_objects, n_attributes, &dataset[0], tree, &predictions[0], attribute_index);

    return Py_BuildValue("O", predictions);
}

// Extension method definition
// Each entry in this list is composed by another list, the later being composed of 4 items:
// ml_name: Method name, as it will be visible to end user; may be different than the intern name defined here;
// ml_meth: Pointer to method implementation;
// ml_flags: Flags with special attributes, such as:
//      Whether it accepts or not parameters, whether it accepts kwarg parameters, etc;
//      If this is a classmethod, a staticmethod, etc;
// ml_doc:  docstring to this function.
static PyMethodDef cpu_methods[] = {
    {"make_predictions", (PyCFunction)make_predictions, METH_VARARGS, make_predictions_doc},
    {"choice", (PyCFunction)choice, METH_VARARGS | METH_KEYWORDS, choice_doc},
    {NULL, NULL, 0, NULL}  // sentinel
};


// module definition
// Arguments in this struct denote extension name, docstring, flags and pointer to extenion's functions.
static struct PyModuleDef cpu_definition = {
    PyModuleDef_HEAD_INIT,  // you should always init the struct with this flag
    "individual", // name of the module
    "Module with predictions for CPU device.", // module documentation
    -1,  // size of per-interpreter state of the module, or -1 if the module keeps state in global variables.
    cpu_methods  // methods of this module
};

// ---------------------------------------------------------------------------------- //
// ---------------------------------------------------------------------------------- //
// ---------------------------------------------------------------------------------- //

// Module inicialization
// Python will call this function when an user imports this extension.
// This function MUST BE NAMED as PyInit_[[name_of_the_module]],
// with name_of_the_module as the EXACT same name as the name entry in script setup.py
PyMODINIT_FUNC PyInit_cpu_device(void) {
    Py_Initialize();
    import_array();  // import numpy arrays
    return PyModule_Create(&cpu_definition);
}


