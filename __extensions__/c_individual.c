#include <Python.h>

static PyObject* make_predictions(PyObject *self, PyObject *args) {
    // list of dict
    // tuple
    // plain dataset (1D)

    int n_objects, n_attributes;

    if (!PyArg_ParseTupleAndKeywords(args, "ii", &n_objects, &n_attributes)) {
      return NULL;
    }

    return Py_BuildValue("s", "Hello, Python extensions!!"); // parse as an string
}

static PyMethodDef c_individual_funcs[] = {
    {"make_predictions", (PyCFunction)make_predictions, METH_VARARGS, "docstring for this functions"},
    {NULL}  // sentinel
};

void initc_individual(void) {
    Py_InitModule3("c_individual", c_individual_funcs, "Extension module example!");
}