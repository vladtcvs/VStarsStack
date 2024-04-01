#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <image_deform_gc.h>

struct ImageDeformGlobalCorrelatorObject
{
    PyObject_HEAD
    struct ImageDeformGlobalCorrelator correlator;
};

extern PyTypeObject ImageDeformGC;
