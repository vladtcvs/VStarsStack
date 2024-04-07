#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <image_deform_lc.h>

struct ImageDeformLocalCorrelatorObject
{
    PyObject_HEAD
    struct ImageDeformLocalCorrelator correlator;
};

extern PyTypeObject ImageDeformLC;
