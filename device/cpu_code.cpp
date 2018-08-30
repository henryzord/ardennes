#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include "numpy/arrayobject.h"

#include <random>
#include <vector>
#include <algorithm>

#define true 1
#define false 0

const char sample_values_doc[] = "sample random values from numpy.random.choice";
static PyObject* sample_values() {

    const int outputSize = 10;

    vector<double> vec(outputSize);

    const vector<double> samples{ 1, 2, 3, 4, 5, 6, 7 };
    const vector<double> probabilities{ 0.1, 0.2, 0.1, 0.5, 0,1 };

    std::default_random_engine generator;
    std::discrete_distribution<int> distribution(probabilities.begin(), probabilities.end());

    vector<int> indices(vec.size());
    std::generate(indices.begin(), indices.end(), [&generator, &distribution]() { return distribution(generator); });

    std::transform(indices.begin(), indices.end(), vec.begin(), [&samples](int index) { return samples[index]; });


    // TODO code below works, don't change yet
    int nd = 2;
    npy_intp dims[] = {3,2};
    PyObject *values = PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
    if(values == NULL) {
        return NULL;
    }
    return Py_BuildValue("O", values);
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
    {"sample_values", (PyCFunction)sample_values, METH_NOARGS, sample_values_doc},
    {NULL, NULL, 0, NULL}  // sentinel
};


// module definition
// Arguments in this struct denote extension name, docstring, flags and pointer to extenion's functions.
static struct PyModuleDef cpu_definition = {
    PyModuleDef_HEAD_INIT,  // you should always init the struct with this flag
    "individual",
    "Module with predictions for CPU device.",
    -1,
    cpu_methods
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


