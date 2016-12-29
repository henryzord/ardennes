//
// Created by henry on 29/12/16.
//

#include <Python.h>

static PyObject *individual_predictor(PyObject *self, PyObject *args) {

//   if (!PyArg_ParseTuple(args, "ids", &i, &d, &s)) {
//      return NULL;
//   }

   /* Do something interesting here. */
   Py_RETURN_NONE;
}

/**
 struct PyMethodDef {
   char *ml_name;  // name in python
   PyCFunction ml_meth;  // name in C
   int ml_flags;  // either METH_VARARGS, METH_KEYWORDS or METH_NOARGS; go figure out the meaning
   char *ml_doc;  // docstring; may be NULL
};
 */

static PyMethodDef module_methods[] = {
   { "__predict_object__", (PyCFunction)individual_predictor, METH_VARARGS, NULL},
   { NULL, NULL, 0, NULL }
};

PyMODINIT_FUNC initIndividual() {
   Py_InitModule3(func, module_methods, "docstring...");
}