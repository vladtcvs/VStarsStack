#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <image_deform.h>

struct ImageDeformObject
{
    PyObject_HEAD
    struct ImageDeform deform;
};

extern PyTypeObject ImageDeform;
