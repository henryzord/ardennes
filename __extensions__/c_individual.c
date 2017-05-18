#include <Python.h>

#define true 1
#define false 0


static int next_node(int current_node, int go_left) {
    return (current_node * 2) + 1 + (!go_left);
}

static void predict_dataset(
    int n_objects, int n_attributes, PyObject *dataset, PyObject *tree, PyObject *predictions,
    PyObject *attribute_index, int multi_tests
    ) {

    int i, terminal, current_node, go_left, int_attr;
    float threshold, val;

    PyObject *list_labels;
    char *label;

    for(i = 0; i < n_objects; i++) {
        current_node = 0;

        while(true) {
            PyObject *node = PyDict_GetItem(tree, Py_BuildValue("i", current_node));
            PyObject *is_terminal = PyDict_GetItemString(node, "terminal");
            terminal = PyObject_IsTrue(is_terminal);

            int is_list = -1;
            list_labels = PyDict_GetItemString(node, "label");

            if(PyObject_TypeCheck(list_labels, &PyList_Type)) {
                is_list = 1;
            } else if((PyObject_TypeCheck(list_labels, &PyString_Type))) {
                label = PyString_AsString(list_labels);
                is_list = 0;
            }

            if(terminal) {
                PyList_SetItem(predictions, i, Py_BuildValue("s", label));
                break;
            } else {
                go_left = 0;

                val = (float)PyFloat_AsDouble(PyList_GetItem(dataset, (Py_ssize_t)(i * n_attributes + int_attr)));

                if(is_list == 0) {
                    int_attr = (int)PyInt_AsLong(PyDict_GetItemString(attribute_index, label));
                    threshold = (float)PyFloat_AsDouble(PyDict_GetItemString(node, "threshold"));
                    go_left += (val <= threshold);
                } else {
                    int index;

                    PyObject *thresholds = PyDict_GetItemString(node, "threshold");

                    for(index = 0; index < multi_tests; index++) {
                        label = PyString_AsString(PyList_GetItem(list_labels, (Py_ssize_t)index));

                        int_attr = (int)PyInt_AsLong(PyDict_GetItemString(attribute_index, label));
                        threshold = (float)PyFloat_AsDouble(PyList_GetItem(thresholds, (Py_ssize_t)index));
                        go_left += (val <= threshold);
                    }
                }
                current_node = next_node(current_node, (go_left > 0));
            }
        }
    }
}

static PyObject* make_predictions(PyObject *self, PyObject *args) {
    int n_objects, n_attributes, multi_tests;
    PyObject *predictions, *tree, *dataset, *attribute_index, *shape;

    if (!PyArg_ParseTuple(
            args, "O!O!O!O!O!i",
            &PyTuple_Type, &shape,
             &PyList_Type, &dataset,
             &PyDict_Type, &tree,
             &PyList_Type, &predictions,
             &PyDict_Type, &attribute_index,
             &multi_tests
    )) {
        return NULL;
    }

    n_objects = (int)PyInt_AsLong(PyTuple_GetItem(shape, 0));
    n_attributes = (int)PyInt_AsLong(PyTuple_GetItem(shape, 1));

    predict_dataset(n_objects, n_attributes, &dataset[0], tree, &predictions[0], attribute_index, multi_tests);

    return Py_BuildValue("O", predictions);
}

const char make_predictions_doc[] = "Makes predictions for a series of unknown data.\n\n"
    ":param shape: shape of dataset.\n"
    ":param dataset: set of unknown data.\n"
    ":param tree: decision tree which will be used to make the predictions.\n"
    ":param predictions: Empty array in which the predictions will be written.\n"
    ":param attribute_index: A dictionary where the attribute names are the keys and the values are their indexes.\n"
    ":param multi_tests: Number of tests used per node.\n"
    ":returns: list of predictions, one entry per object passed.";

static PyMethodDef c_individual_funcs[] = {
    {"make_predictions", (PyCFunction)make_predictions, METH_VARARGS, &make_predictions_doc[0]},
    {NULL}  // sentinel
};

void initc_individual(void) {
    Py_InitModule3("c_individual", c_individual_funcs, "Extension module example!");
}
