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

    PyObject *node, *list_labels, *thresholds;

    for(int i = 0; i < n_objects; i++) {
        int current_node = 0;

        while(true) {
            node = PyDict_GetItem(tree, Py_BuildValue("i", current_node));
            int terminal = PyObject_IsTrue(PyDict_GetItemString(node, "terminal"));

            list_labels = PyDict_GetItemString(node, "label");

            if(terminal) {
                PyList_SetItem(predictions, i, Py_BuildValue("s", PyString_AsString(list_labels)));
                break;
            } else {
                int go_left = 0;

                thresholds = PyDict_GetItemString(node, "threshold");

                for(int index = 0; index < multi_tests; index++) {
                    char *label = PyString_AsString(
                        PyList_GetItem(list_labels, index)
                    );
                    float threshold = (float)PyFloat_AsDouble(
                        PyList_GetItem(thresholds, index)
                    );
                    int int_attr = (int)PyInt_AsLong(PyDict_GetItemString(attribute_index, label));
                    float val = (float)PyFloat_AsDouble(PyList_GetItem(dataset, (i * n_attributes + int_attr)));

                    go_left += (val <= threshold);
                }
                current_node = next_node(current_node, (go_left > (multi_tests/2)));
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
