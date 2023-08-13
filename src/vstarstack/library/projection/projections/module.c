/*
 * Copyright (c) 2023 Vladislav Tsendrovskii
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

#include <stdio.h>

#include "projection_module.h"

#define BASENAME "vstarstack.library.projection.projections"

// arguments: W, H, F, w, h
static int PerspectiveProjection_init(PyObject *_self, PyObject *args, PyObject *kwds)
{
    double W, H, F;
    int w, h;
    struct PerspectiveProjectionObject *self = (struct PerspectiveProjectionObject *)_self;
    static char *kwlist[] = {"w", "h", "W", "H", "F", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiddd", kwlist,
                                     &w, &h,
                                     &W, &H, &F))
        return -1;

    if (!perspective_projection_init(&self->proj, W, H, F, w, h))
        return -1;

    return 0;
}

static int OrthographicProjection_init(PyObject *_self, PyObject *args, PyObject *kwds)
{
    double a, b, angle, rot;
    int w, h;
    struct OrthographicProjectionObject *self = (struct OrthographicProjectionObject *)_self;
    static char *kwlist[] = {"w", "h", "a", "b", "angle", "rot", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iidddd", kwlist,
                                     &w, &h,
                                     &a, &b, &angle, &rot))
        return -1;
    
    if (!orthographic_projection_init(&self->proj, w, h, a, b, angle, rot))
        return -1;

    return 0;
}

static int EquirectangularProjection_init(PyObject *_self, PyObject *args, PyObject *kwds)
{
    int w, h;
    struct EquirectangularProjectionObject *self = (struct EquirectangularProjectionObject *)_self;
    static char *kwlist[] = {"w", "h", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ii", kwlist,
                                     &w, &h))
        return -1;
    
    if (!equirectangular_projection_init(&self->proj, w, h))
        return -1;

    return 0;
}

// Common methods

static PyObject *Projection_project(void *proj,
                                    PyObject *args,
                                    PyObject *kwds,
                                    forward_project_f function)
{
    double x, y;
    double lon, lat;
    static char *kwlist[] = {"x", "y", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "dd", kwlist,
                                     &x, &y))
        return Py_BuildValue("(OO)", Py_None, Py_None);

    if (!function(proj, y, x, &lat, &lon))
        return Py_BuildValue("(OO)", Py_None, Py_None);
    return Py_BuildValue("(dd)", lon, lat);
}

static PyObject *Projection_reverse(void *proj,
                                    PyObject *args,
                                    PyObject *kwds,
                                    reverse_project_f function)
{
    double x, y;
    double lon, lat;
    static char *kwlist[] = {"lon", "lat", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "dd", kwlist,
                                     &lon, &lat))
        return Py_BuildValue("(OO)", Py_None, Py_None);

    //printf("lat %lf lon %lf\n", lat, lon);
    if (!function(proj, lat, lon, &y, &x))
        return Py_BuildValue("(OO)", Py_None, Py_None);
    //printf("x %lf y %lf\n", x, y);
    return Py_BuildValue("(dd)", x, y);
}

// Perspective

static PyObject *Perspective_forward(PyObject *_self,
                                     PyObject *args,
                                     PyObject *kwds)
{
    struct PerspectiveProjectionObject *self = (struct PerspectiveProjectionObject *)_self;
    return Projection_project(&self->proj, args, kwds, perspective_projection_project);
}


static PyObject *Perspective_reverse(PyObject *_self,
                                     PyObject *args,
                                     PyObject *kwds)
{
    struct PerspectiveProjectionObject *self = (struct PerspectiveProjectionObject *)_self;
    return Projection_reverse(&self->proj, args, kwds, perspective_projection_reverse);
}

static PyMethodDef _PerspectiveProjection_methods[] = {
    {"project", (PyCFunction)Perspective_forward, METH_VARARGS | METH_KEYWORDS,
     "Project x,y to lat,lon"},
    {"reverse", (PyCFunction)Perspective_reverse, METH_VARARGS | METH_KEYWORDS,
     "Project lon,lat to x,y"},
    {NULL} /* Sentinel */
};

static PyTypeObject PerspectiveProjection = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = BASENAME ".PerspectiveProjection",
    .tp_doc = PyDoc_STR("Perspective projection object"),
    .tp_basicsize = sizeof(struct PerspectiveProjectionObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = PerspectiveProjection_init,
    .tp_methods = _PerspectiveProjection_methods,
};


// Orthographic

static PyObject *Orthographic_forward(PyObject *_self,
                                      PyObject *args,
                                      PyObject *kwds)
{
    struct OrthographicProjectionObject *self = (struct OrthographicProjectionObject *)_self;
    return Projection_project(&self->proj, args, kwds, orthographic_projection_project);
}

static PyObject *Orthographic_reverse(PyObject *_self,
                                      PyObject *args,
                                      PyObject *kwds)
{
    struct OrthographicProjectionObject *self = (struct OrthographicProjectionObject *)_self;
    return Projection_reverse(&self->proj, args, kwds, orthographic_projection_reverse);
}

static PyMethodDef _OrthographicProjection_methods[] = {
    {"project", (PyCFunction)Orthographic_forward, METH_VARARGS | METH_KEYWORDS,
     "Project x,y to lon,lat"},
    {"reverse", (PyCFunction)Orthographic_reverse, METH_VARARGS | METH_KEYWORDS,
     "Project lon,lat to x,y"},
    {NULL} /* Sentinel */
};

static PyTypeObject OrthographicProjection = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = BASENAME ".OrthographicProjection",
    .tp_doc = PyDoc_STR("Orthographic projection object"),
    .tp_basicsize = sizeof(struct OrthographicProjectionObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = OrthographicProjection_init,
    .tp_methods = _OrthographicProjection_methods,
};


// Equirectangular

static PyObject *Equirectangular_forward(PyObject *_self,
                                         PyObject *args,
                                         PyObject *kwds)
{
    struct EquirectangularProjectionObject *self = (struct EquirectangularProjectionObject *)_self;
    return Projection_project(&self->proj, args, kwds, equirectangular_projection_project);
}

static PyObject *Equirectangular_reverse(PyObject *_self,
                                         PyObject *args,
                                         PyObject *kwds)
{
    struct EquirectangularProjectionObject *self = (struct EquirectangularProjectionObject *)_self;
    return Projection_reverse(&self->proj, args, kwds, equirectangular_projection_reverse);
}

static PyMethodDef _EquirectangularProjection_methods[] = {
    {"project", (PyCFunction)Equirectangular_forward, METH_VARARGS | METH_KEYWORDS,
     "Project x,y to lon,lat"},
    {"reverse", (PyCFunction)Equirectangular_reverse, METH_VARARGS | METH_KEYWORDS,
     "Project lon,lat to x,y"},
    {NULL} /* Sentinel */
};

static PyTypeObject EquirectangularProjection = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = BASENAME ".EquirectangularProjection",
    .tp_doc = PyDoc_STR("Equirectangular projection object"),
    .tp_basicsize = sizeof(struct EquirectangularProjectionObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = EquirectangularProjection_init,
    .tp_methods = _EquirectangularProjection_methods,
};

// Module

static PyModuleDef projectionModule = {
    PyModuleDef_HEAD_INIT,
    .m_name = BASENAME,
    .m_doc = "Projection module",
    .m_size = -1,
};

PyMODINIT_FUNC
PyInit_projections(void)
{
    PyObject *m = PyModule_Create(&projectionModule);
    if (m == NULL)
        return NULL;

    if (PyType_Ready(&PerspectiveProjection) < 0)
    {
        fprintf(stderr, "Bad type _PerspectiveProjection\n");
        return NULL;
    }

    if (PyType_Ready(&OrthographicProjection) < 0)
    {
        fprintf(stderr, "Bad type _OrthographicProjection\n");
        return NULL;
    }

    if (PyType_Ready(&EquirectangularProjection) < 0)
    {
        fprintf(stderr, "Bad type _EquirectangularProjection\n");
        return NULL;
    }

    Py_INCREF(&PerspectiveProjection);
    if (PyModule_AddObject(m, "PerspectiveProjection", (PyObject *)&PerspectiveProjection) < 0)
    {
        Py_DECREF(&PerspectiveProjection);
        Py_DECREF(m);
        fprintf(stderr, "Can not add PerspectiveProjection\n");
        return NULL;
    }

    Py_INCREF(&OrthographicProjection);
    if (PyModule_AddObject(m, "OrthographicProjection", (PyObject *)&OrthographicProjection) < 0)
    {
        Py_DECREF(&PerspectiveProjection);
        Py_DECREF(&OrthographicProjection);
        Py_DECREF(m);
        fprintf(stderr, "Can not add OrthographicProjection\n");
        return NULL;
    }

    Py_INCREF(&EquirectangularProjection);
    if (PyModule_AddObject(m, "EquirectangularProjection", (PyObject *)&EquirectangularProjection) < 0)
    {
        Py_DECREF(&PerspectiveProjection);
        Py_DECREF(&OrthographicProjection);
        Py_DECREF(&EquirectangularProjection);
        Py_DECREF(m);
        fprintf(stderr, "Can not add EquirectangularProjection\n");
        return NULL;
    }

    return m;
}
