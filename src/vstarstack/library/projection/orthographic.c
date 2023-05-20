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
    int w;        // image width
    int h;        // image height
    double a;     // planet ellipse major axis
    double b;     // planet ellipse minor axis
    double angle; // planet ellipse slope
    double rot;   // planet rotation angle
};

// arguments: W, H, F, w, h
static int Projection_init(PyObject *_self, PyObject *args, PyObject *kwds)
{
    struct ProjectionObject *self = (struct ProjectionObject *)_self;
    static char *kwlist[] = {"w", "h", "a", "b", "angle", "rot", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iidddd", kwlist,
                                     &self->w, &self->h, &self->a, &self->b,
                                     &self->angle, &self->rot))
        return -1;
    if (self->h <= 0 || self->w <= 0 || self->a <= 0 || self->b <= 0)
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
    
    x -= self->w/2;
    y -= self->h/2;
    double X = x*cos(self->angle) - y*sin(self->angle);
    double Y = x*sin(self->angle) + y*cos(self->angle);

    double sin_lat = Y / (self->b/2);
    if (fabs(sin_lat) > 1)
    {
        Py_INCREF(Py_None);
        Py_INCREF(Py_None);
        return Py_BuildValue("(OO)", Py_None, Py_None);
    }
    double lat = -asin(sin_lat);
    double sin_lon = X / (self->a/2) / cos(lat);
    if (fabs(sin_lon) > 1)
    {
        Py_INCREF(Py_None);
        Py_INCREF(Py_None);
        return Py_BuildValue("(OO)", Py_None, Py_None);
    }

    double lon = asin(sin_lon) + self->rot;
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

    double x = self->a/2 * sin(lon - self->rot) * cos(lat);
    double z = -self->b/2 * sin(lat);
    double X = x*cos(self->angle) + z*sin(self->angle) + self->w/2;
    double Y = -x*sin(self->angle) + z*cos(self->angle) + self->h/2;
    return Py_BuildValue("(dd)", Y, X);
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
    .tp_name = "vstarstack.library.projection.orthographic.Projection",
    .tp_doc = PyDoc_STR("Orthographic projection object"),
    .tp_basicsize = sizeof(struct ProjectionObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = Projection_init,
    .tp_methods = Projection_methods,
};

static PyModuleDef orthographicModule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "vstarstack.library.projection.orthographic",
    .m_doc = "Orthographic projection module",
    .m_size = -1,
};

PyMODINIT_FUNC
PyInit_orthographic(void)
{
    PyObject *m;
    if (PyType_Ready(&Projection) < 0)
        return NULL;

    m = PyModule_Create(&orthographicModule);
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
