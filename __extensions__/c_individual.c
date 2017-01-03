#include <Python.h>

#define true 1
#define false 0


static void predict_dataset(int n_objects, int n_attributes, PyObject *dataset, PyObject *predictions, PyObject *tree) {
    int i, terminal, current_node, go_left;

//    printf("some number: %d\n", (int)PyInt_AsLong(PyList_GetItem(predictions, 0)));

    for(i = 0; i < n_objects; i++) {
        current_node = 0; go_left = 0;

        while(true) {
            PyObject *node = PyDict_GetItem(tree, Py_BuildValue("i", 0));
            PyObject *is_terminal = PyDict_GetItemString(node, "terminal");
            int terminal = PyObject_IsTrue(is_terminal);

            if(terminal) {  // if is terminal
                break;
//                char *label =

//                PyList_SetItem(predictions, i, Py_BuildValue("s", terminal));
//                predictions[i] = (int)thresholds[current_node];
//                break;
            } else {
//                PyList_GetItem(dataset, i * n_objects + )


                go_left = dataset[i * n_objects + attribute_index[current_node]] <= thresholds[current_node];
                current_node = next_node(current_node, go_left, finished);
            }
            break;
        }
    }
}

static PyObject* make_predictions(PyObject *self, PyObject *args) {
    PyObject *predictions, *tree, *dataset;
    int n_objects, n_attributes;

    if (!PyArg_ParseTuple(args, "iiOOO!", &n_objects, &n_attributes, &dataset, &tree, &PyList_Type, &predictions)) {
        return NULL;
    }

    predict_dataset(n_objects, n_attributes, &dataset[0], &predictions[0], tree);

    return Py_BuildValue("O", predictions);
}

static PyMethodDef c_individual_funcs[] = {
    {"make_predictions", (PyCFunction)make_predictions, METH_VARARGS, "docstring for this function"},
    {NULL}  // sentinel
};

void initc_individual(void) {
    Py_InitModule3("c_individual", c_individual_funcs, "Extension module example!");
}
