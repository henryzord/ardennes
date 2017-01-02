#include <Python.h>

static void predict_dataset(int n_objects, int n_attributes, PyFloatObject *dataset, PyObject *predictions, PyObject *tree) {
    int i;
//    int i, h, current_node, go_left, finished;

    Py_Initialize();
    for(i = 0; i < n_objects; i++) {
//        current_node = 0;
//        finished = 0;

        predictions[i] = *Py_BuildValue("i", 0);
        break;
//        while(true) {
//            if((bool)PyDict_GetItem(tree, "terminal")) {  // if is terminal
//                predictions[i] = (int)thresholds[current_node];
//                break;
//            } else {
//                go_left = dataset[i * n_objects + attribute_index[current_node]] <= thresholds[current_node];
//                current_node = next_node(current_node, go_left, finished);
//            }
//        }
    }
    Py_Finalize();
}

static PyObject* make_predictions(PyObject *self, PyObject *args) {
    PyObject *dataset, *predictions, *tree;
    int n_objects, n_attributes;

    if (!PyArg_ParseTuple(args, "iiOOO", &n_objects, &n_attributes, &dataset, &predictions, &tree)) {
        return NULL;
    }

    predict_dataset(n_objects, n_attributes, &((PyFloatObject*)dataset)[0], predictions, tree);

    return Py_BuildValue("s", "Hello, Python extensions!!"); // returns an string
}

static PyMethodDef c_individual_funcs[] = {
    {"make_predictions", (PyCFunction)make_predictions, METH_VARARGS, "docstring for this function"},
    {NULL}  // sentinel
};

void initc_individual(void) {
    Py_InitModule3("c_individual", c_individual_funcs, "Extension module example!");
}
