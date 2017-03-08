#include <Python.h>

#define true 1
#define false 0


static int next_node(int current_node, int go_left) {
    return (current_node * 2) + 1 + (!go_left);
}

static void predict_dataset(int n_objects, int n_attributes, PyObject *dataset, PyObject *tree, PyObject *predictions, PyObject *attribute_index) {
    int i, terminal, current_node, go_left, int_attr;
    char *label;
    float threshold, val;

    for(i = 0; i < n_objects; i++) {
        current_node = 0;

        while(true) {
            PyObject *node = PyDict_GetItem(tree, Py_BuildValue("i", current_node));
            PyObject *is_terminal = PyDict_GetItemString(node, "terminal");
            label = PyString_AsString(PyDict_GetItemString(node, "label"));

            terminal = PyObject_IsTrue(is_terminal);

            if(terminal) {
                PyList_SetItem(predictions, i, Py_BuildValue("s", label));
                break;
            } else {
                int_attr = (int)PyInt_AsLong(PyDict_GetItemString(attribute_index, label));
                threshold = (float)PyFloat_AsDouble(PyDict_GetItemString(node, "threshold"));
                val = (float)PyFloat_AsDouble(PyList_GetItem(dataset, i * n_attributes + int_attr));

                go_left = val <= threshold;
                current_node = next_node(current_node, go_left);
            }
        }
    }
}

static PyObject* make_predictions(PyObject *self, PyObject *args) {
    int n_objects, n_attributes;
    PyObject *predictions, *tree, *dataset, *attribute_index, *shape;

    if (!PyArg_ParseTuple(
            args, "O!O!O!O!O!",
            &PyTuple_Type, &shape,
            &PyList_Type, &dataset,
            &PyDict_Type, &tree,
            &PyList_Type, &predictions,
            &PyDict_Type, &attribute_index)) {
        return NULL;
    }

    n_objects = (int)PyInt_AsLong(PyTuple_GetItem(shape, 0));
    n_attributes = (int)PyInt_AsLong(PyTuple_GetItem(shape, 1));

    predict_dataset(n_objects, n_attributes, &dataset[0], tree, &predictions[0], attribute_index);

    return Py_BuildValue("O", predictions);
}

const char make_predictions_doc[] = "Makes predictions for a series of unknown data.\n\n"
    ":param shape: Shape of Dataset.\n"
    ":param dataset: set of unknown data.\n"
    ":param tree: decision tree which will be used to make the predictions.\n"
    ":param predictions: Empty array in which the predictions will be written.\n"
    "attribute_index: A dictionary where the attribute names are the keys and the values are their indexes.";

static PyMethodDef c_individual_funcs[] = {
    {"make_predictions", (PyCFunction)make_predictions, METH_VARARGS, &make_predictions_doc[0]},
    {NULL}  // sentinel
};

void initc_individual(void) {
    Py_InitModule3("c_individual", c_individual_funcs, "Extension module example!");
}
