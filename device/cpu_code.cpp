#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <random>
#include <cmath>

#include "cpu_utils.h"


float entropy_by_index(PyArrayObject *dataset, PyArrayObject *subset_index, int n_classes) {
    npy_intp *dims = PyArray_DIMS((PyArrayObject*)dataset);
    int n_objects = dims[0], n_attributes = dims[1];

    const int class_index = n_attributes - 1;
    float count_class, entropy = 0, subset_size = 0;

    for(int i = 0; i < n_objects; i++) {
        subset_size += (int)PyLong_AsLong(at(subset_index, i));
    }

    for(int k = 0; k < n_classes; k++) {
        count_class = 0;
        for(int i = 0; i < n_objects; i++) {
            count_class += (float)PyFloat_AsDouble(at(subset_index, i)) * (k == (float)PyFloat_AsDouble(at(dataset, i, class_index)));
        }
        entropy += select((count_class / subset_size) * std::log2(count_class / subset_size), (float) 0,
                          count_class <= 0);
    }

    return -entropy;
}

float device_gain_ratio(PyArrayObject *dataset, PyArrayObject *subset_index,
    int attribute_index, float candidate, int n_classes, float subset_entropy) {

    npy_intp *dataset_dims = PyArray_DIMS(dataset);
    int n_objects = (int)dataset_dims[0], n_attributes = (int)dataset_dims[1];

    const int class_index = n_attributes - 1;


    float   left_entropy = 0, right_entropy = 0,
            left_branch_size = 0, right_branch_size = 0,
            left_count_class, right_count_class,
            subset_size = 0, sum_term, is_from_class;

    bool is_left;

    for(int k = 0; k < n_classes; k++) {
        left_count_class = 0; right_count_class = 0;
        left_branch_size = 0; right_branch_size = 0;
        subset_size = 0;

        for(int i = 0; i < n_objects; i++) {
            float val = (float)PyFloat_AsDouble(at(subset_index, i));

            is_left = (bool)(PyFloat_AsDouble(at(dataset, i, attribute_index)) <= candidate);
            is_from_class = (float)(std::fabs((float)PyFloat_AsDouble(at(dataset, i, class_index)) - k) < 0.01);

            left_branch_size += (float)is_left * val;
            right_branch_size += (float)(!is_left) * val;

            left_count_class += (float)is_left * val * is_from_class;
            right_count_class += (float)(!is_left) * val * is_from_class;

            subset_size += val;
        }
        left_entropy += select(
                (left_count_class / left_branch_size) * std::log2(left_count_class / left_branch_size),
                (float) 0,
                left_count_class <= 0
        );
        right_entropy += select(
                (right_count_class / right_branch_size) * std::log2(right_count_class / right_branch_size),
                (float) 0,
                right_count_class <= 0
        );
    }

    sum_term =
        ((left_branch_size / subset_size) * -left_entropy) +
        ((right_branch_size / subset_size) * -right_entropy);

    float
        info_gain = subset_entropy - sum_term,
        split_info = -(
            select((left_branch_size / subset_size) * std::log2(left_branch_size / subset_size), (float) 0,
                   left_branch_size <= 0) +
                    select((right_branch_size / subset_size) * std::log2(right_branch_size / subset_size), (float) 0,
                           right_branch_size <= 0)
        );

    return select(info_gain / split_info, (float) 0, split_info <= 9e-7);
}

const char gain_ratio_doc[] = "Calculates gain ratio of a series of thresholds.";

static PyObject* gain_ratio(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *dataset, *candidates, *subset_index;
    int n_classes, attribute_index;

    static char *kwds[] = {
        "dataset", "subset_index", "attribute_index", "candidates", "n_classes", NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOiOi", kwds,
        &dataset, &subset_index, &attribute_index, &candidates, &n_classes)) {
        return NULL;
    }

    static char *obj_names[] = {"dataset", "subset_index", "candidates", NULL};
    static int obj_dtypes[] = {NPY_FLOAT32, NPY_BOOL, NPY_FLOAT32, -1};
    PyObject *objs[] = {dataset, subset_index, candidates, NULL};

    if(!perform_checks(obj_names, obj_dtypes, objs)) {
        return NULL;
    }

    int n_candidates = (int)PyArray_SIZE((PyArrayObject*)candidates);

    // entropy for whole subset provided, not only for left and right branches
    float subset_entropy = entropy_by_index((PyArrayObject*)dataset, (PyArrayObject*)subset_index, n_classes);

    npy_intp gain_ratios_dims[] = {(npy_intp)n_candidates};
    PyArrayObject *gain_ratios = (PyArrayObject*)PyArray_SimpleNew(1, gain_ratios_dims, NPY_FLOAT32);
    npy_intp gain_ratios_itemsize = PyArray_ITEMSIZE(gain_ratios);
    char *gain_ratios_data = PyArray_BYTES(gain_ratios);

    for(int idx = 0; idx < n_candidates; idx++) {
        float local_gain_ratio = device_gain_ratio(
            (PyArrayObject*)dataset, (PyArrayObject*)subset_index, attribute_index,
            (float)PyFloat_AsDouble(at((PyArrayObject*)candidates, idx)), n_classes, subset_entropy
        );

        PyArray_SETITEM(gain_ratios, gain_ratios_data, Py_BuildValue("f", local_gain_ratio));
        gain_ratios_data += gain_ratios_itemsize;
    }

    return Py_BuildValue("O", gain_ratios);
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

    int sampled_ndims = -1;
    npy_intp *sampled_dims;
    if(size != NULL) {
        if(PyTuple_Check(size)) {
            sampled_ndims = (int)PyTuple_Size(size);
            sampled_dims = new npy_intp [sampled_ndims];

            for(int i = 0; i < sampled_ndims; i++) {
                sampled_dims[i] = (int)PyLong_AsLong(PyTuple_GetItem(size, i));
            }
        } else if (PyLong_Check(size)) {
            sampled_ndims = 1;
            sampled_dims = new npy_intp [1];
            sampled_dims[0] = (int)PyLong_AsLong(size);
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

    PyArray_Descr *a_descr = PyArray_DESCR((PyArrayObject*)a);
    PyObject *sampled_obj = PyArray_NewFromDescr(
        &PyArray_Type, a_descr,
        sampled_ndims, sampled_dims, NULL, NULL, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE, NULL
    );

    PyArrayObject *sampled = (PyArrayObject*)sampled_obj;

    if(sampled == NULL) {
        PyErr_SetString(PyExc_ValueError, "Exception ocurred while trying to allocate space for numpy array");
        return NULL;
    }

    // type checking ends here; real code begins here

    npy_intp sampled_size = PyArray_SIZE(sampled),
             a_itemsize = PyArray_ITEMSIZE((PyArrayObject*)a),
             p_itemsize = PyArray_ITEMSIZE((PyArrayObject*)p);

    int *sampled_counters = (int*)malloc(sizeof(int) * sampled_ndims);
    for(int j = 0; j < sampled_ndims; j++) {
        sampled_counters[j] = 0;
    }

    char *sampled_ptr = PyArray_BYTES(sampled), *p_ptr, *a_ptr;  // data pointers
    npy_intp sampled_itemsize = PyArray_ITEMSIZE(sampled);

    int num, div;
    float sum, spread = 1000, p_data;
    PyObject *a_data;

    for(int i = 0; i < sampled_size; i++) {
        num = rand() % (int)spread;  // random sampled number
        sum = 0;  // sum of probabilities so far

        p_ptr = PyArray_BYTES((PyArrayObject*)p);
        a_ptr = PyArray_BYTES((PyArrayObject*)a);

        for(int k = 0; k < a_dims[0]; k++) {
            p_data = (float)PyFloat_AsDouble(PyArray_GETITEM((PyArrayObject*)p, p_ptr));
            a_data = PyArray_GETITEM((PyArrayObject*)a, a_ptr);
            p_ptr += p_itemsize;
            a_ptr += a_itemsize;

            div = (int)(num/((sum + p_data) * spread));

            if(div <= 0) {
                PyArray_SETITEM(sampled, sampled_ptr, a_data);
                break;
            }
            sum += p_data;
        }
        sampled_ptr += sampled_itemsize;
    }

    return Py_BuildValue("O", sampled);
}

static int next_node(int current_node, int go_left) {
    return (current_node * 2) + 1 + (!go_left);
}

const char make_predictions_doc[] = "Makes predictions for a series of unknown data.\n";
static PyObject* make_predictions(PyObject *self, PyObject *args, PyObject *kwargs) {

    static char *kwds[] = {"full_df", "attr_index", "nodes", "threshold", "terminal", "predictions", NULL};
    PyObject *full_df, *attr_index, *nodes, *threshold, *terminal, *predictions;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OO!OOOO", kwds, &full_df, &PyDict_Type, &attr_index, &nodes, &threshold, &terminal, &predictions)) {
        return NULL;
    }

    static char *obj_names[] = {"full_df", "nodes", "threshold", "terminal", "predictions", NULL};
    static int obj_dtypes[] = {NPY_FLOAT32, NPY_UNICODE, NPY_FLOAT32, NPY_BOOL, NPY_UNICODE, -1};
    PyObject *objs[] = {full_df, nodes, threshold, terminal, predictions, NULL};

    if(!perform_checks(obj_names, obj_dtypes, objs)) {
        return NULL;
    }

    npy_intp *full_df_dims = PyArray_DIMS((PyArrayObject*)full_df);
    int n_objects = (int)full_df_dims[0];

    char *predictions_data = PyArray_BYTES((PyArrayObject*)predictions);
    npy_intp predictions_itemsize = PyArray_ITEMSIZE((PyArrayObject*)predictions);

    int int_attr, current_node;
    bool is_terminal;
    float current_threshold, value;

    for(int i = 0; i < n_objects; i++) {
        current_node = 0;

        while(true) {
            is_terminal = (bool)PyLong_AsLong(at((PyArrayObject*)terminal, current_node));
            PyObject *label_obj = PyUnicode_AsEncodedString(at((PyArrayObject*)nodes, current_node), "utf-8", "Error ~");
            char *label_char =  PyBytes_AsString(label_obj);

            if(is_terminal) {
                PyArray_SETITEM((PyArrayObject*)predictions, predictions_data, label_obj);
                predictions_data += predictions_itemsize;
                break;
            }  else {
                current_threshold = (float)PyFloat_AsDouble(at((PyArrayObject*)threshold, current_node));
                PyObject *partial = PyDict_GetItemString(attr_index, label_char);

                int_attr = (int)PyLong_AsLong(partial);
                value = (float)PyFloat_AsDouble(at((PyArrayObject*)full_df, i, int_attr));
                current_node = next_node(current_node, (value <= current_threshold));
            }
        }
    }
    return predictions;
}

/**
 * Extension method definition
 * Each entry in this list is composed by another list, the later being composed of 4 items:
 * ml_name: Method name, as it will be visible to end user; may be different than the intern name defined here;
 * ml_meth: Pointer to method implementation;
 * ml_flags: Flags with special attributes, such as:
 *     Whether it accepts or not parameters, whether it accepts kwarg parameters, etc;
 *     If this is a classmethod, a staticmethod, etc;
 * ml_doc:  docstring to this function.
 */
static PyMethodDef cpu_methods[] = {
    {"make_predictions", (PyCFunction)make_predictions, METH_VARARGS | METH_KEYWORDS, make_predictions_doc},
    {"choice", (PyCFunction)choice, METH_VARARGS | METH_KEYWORDS, choice_doc},
    {"gain_ratio", (PyCFunction)gain_ratio, METH_VARARGS | METH_KEYWORDS, gain_ratio_doc},
    {NULL, NULL, 0, NULL}  // sentinel
};


// module definition
// Arguments in this struct denote extension name, docstring, flags and pointer to extenion's functions.
static struct PyModuleDef cpu_definition = {
    PyModuleDef_HEAD_INIT,  // you should always init the struct with this flag
    "individual", // name of the module
    "Module with methods for decision trees: prediction of instances and information gain calculation.", // module documentation
    -1,  // size of per-interpreter state of the module, or -1 if the module keeps state in global variables.
    cpu_methods  // methods of this module
};

// ---------------------------------------------------------------------------------- //
// ---------------------------------------------------------------------------------- //
// ---------------------------------------------------------------------------------- //

// Module initialization
// Python will call this function when an user imports this extension.
// This function MUST BE NAMED as PyInit_[[name_of_the_module]],
// with name_of_the_module as the EXACT same name as the name entry in script setup.py
PyMODINIT_FUNC PyInit_cpu_device(void) {
    Py_Initialize();
    import_array();  // import numpy arrays
    return PyModule_Create(&cpu_definition);
}
