#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <image_grid.h>

struct ImageGridObject
{
    PyObject_HEAD
    struct ImageGrid grid;
};

extern PyTypeObject ImageGrid;
