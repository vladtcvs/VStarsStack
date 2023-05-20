/*
 * Copyright (c) 2022 Vladislav Tsendrovskii
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#include <math.h>
#include <stdio.h>

struct ProjectionObject
{
    PyObject_HEAD
    int h;          // height of image in pixels
    int w;          // width of image in pixels
};

// arguments: w, h
static int Projection_init(PyObject *_self, PyObject *args, PyObject *kwds)
{
    struct ProjectionObject *self = (struct ProjectionObject *)_self;
    static char *kwlist[] = {"w", "h", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ii", kwlist,
                                     &self->w, &self->h))
        return -1;
    if (self->h <= 0 || self->w <= 0)
        return -1;

    return 0;
}

static PyObject *Projection_project(PyObject *_self, PyObject *args, PyObject *kwds)
{
    struct ProjectionObject *self = (struct ProjectionObject *)_self;
    double x, y;
    static char *kwlist[] = {"y", "x", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "dd", kwlist,
                                     &y, &x))
        return Py_None;
    double lon = (1 - 2*x/self->w) * M_PI;
    double lat = (1 - 2*y/self->h) * M_PI_2;

    return Py_BuildValue("(dd)", lat, lon);
}

static PyObject *Projection_reverse(PyObject *_self, PyObject *args, PyObject *kwds)
{
    struct ProjectionObject *self = (struct ProjectionObject *)_self;
    double lat, lon;
    static char *kwlist[] = {"lat", "lon", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "dd", kwlist,
                                     &lat, &lon))
        return Py_None;

    double x = (1 - lon / M_PI)/2*self->w;
    double y = (1 - lat / M_PI_2)/2*self->h;
    return Py_BuildValue("(dd)", y, x);
}

static PyMethodDef Projection_methods[] = {
    {"project", (PyCFunction)Projection_project, METH_VARARGS | METH_KEYWORDS,
     "Project y,x to lat,lon"},
    {"reverse", (PyCFunction)Projection_reverse, METH_VARARGS | METH_KEYWORDS,
     "Project lat,lon to y,x"},
    {NULL} /* Sentinel */
};

static PyTypeObject Projection = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "vstarstack.library.projection.equirectangular.Projection",
    .tp_doc = PyDoc_STR("Equirectangular projection object"),
    .tp_basicsize = sizeof(struct ProjectionObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = Projection_init,
    .tp_methods = Projection_methods,
};

static PyModuleDef equirectangularModule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "vstarstack.library.projection.equirectangular",
    .m_doc = "Equirectangular projection module",
    .m_size = -1,
};

PyMODINIT_FUNC
PyInit_equirectangular(void)
{
    PyObject *m;
    if (PyType_Ready(&Projection) < 0)
        return NULL;

    m = PyModule_Create(&equirectangularModule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&Projection);
    if (PyModule_AddObject(m, "Projection", (PyObject *)&Projection) < 0)
    {
        Py_DECREF(&Projection);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
