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


#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#include <numpy/ndarraytypes.h>
#include <numpy/ndarrayobject.h>
#include <stdlib.h>

#include "projection_module.h"

#include "lib/sphere.h"
#include "lib/flat.h"

#define BASENAME "vstarstack.library.movement.movements"

struct SphereMovementObject
{
    PyObject_HEAD
    struct SphereMovement mov;
};

static bool generic_forward_project(void *self,
                                    double y, double x,
                                    double *lat, double *lon)
{
    PyObject *projection = (PyObject *)self;
    PyObject *res = PyObject_CallMethod(projection, "project", "(dd)", x, y);
    PyObject *_lat = PyTuple_GetItem(res, 1);
    PyObject *_lon = PyTuple_GetItem(res, 0);
    if (_lat == NULL || _lon == NULL)
        return false;
    *lat = PyFloat_AsDouble(_lat);
    *lon = PyFloat_AsDouble(_lon);
    //printf("xy %lf:%lf -> lonlat %lf:%lf\n", x, y, *lon, *lat);
    return true;
}

static bool generic_reverse_project(void *self,
                                    double lat, double lon,
                                    double *y, double *x)
{
    PyObject *projection = (PyObject *)self;
    PyObject *res = PyObject_CallMethod(projection, "reverse", "(dd)", lon, lat);
    PyObject *_y = PyTuple_GetItem(res, 1);
    PyObject *_x = PyTuple_GetItem(res, 0);
    if (_y == NULL || _x == NULL)
        return false;
    *y = PyFloat_AsDouble(_y);
    *x = PyFloat_AsDouble(_x);
    //printf("lonlat %lf:%lf -> xy %lf:%lf\n", lon, lat, *x, *y);
    return true;
}

static int SphereMovements_init(PyObject *_self, PyObject *args, PyObject *kwds)
{
    double w, x, y, z;
    struct SphereMovementObject *self = (struct SphereMovementObject *)_self;
    static char *kwlist[] = {"w", "x", "y", "z", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "dddd", kwlist,
                                     &w, &x, &y, &z))
    {
        return -1;
    }

    struct quat q = {
        .w = w,
        .x = x,
        .y = y,
        .z = z,
    };

    sphere_movement_init(&self->mov, q);
    return 0;
}

bool is_perspective(PyObject *obj)
{
    return false;
}

typedef void (*action_f)(struct SphereMovement *mov,
                         const double *posi, double *poso, size_t num,
                         const struct ProjectionDef *in_proj,
                         const struct ProjectionDef *out_proj);

static PyObject *apply_action(PyObject *_self,
                              PyObject *args,
                              PyObject *kwds,
                              action_f fun)
{
    PyObject *input_proj, *output_proj;
    PyArrayObject *points;
    struct SphereMovementObject *self = (struct SphereMovementObject *)_self;

    static char *kwlist[] = {"points", "input_projection", "output_projection", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOO", kwlist,
                                     &points, &input_proj, &output_proj))
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments");
        Py_INCREF(Py_None);
        return Py_None;
    }

    if (PyArray_TYPE(points) != NPY_DOUBLE)
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - should be dtype == double");
        Py_INCREF(Py_None);
        return Py_None;
    }

    if (PyArray_NDIM(points) != 2)
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - should be len(shape) == 2");
        Py_INCREF(Py_None);
        return Py_None;
    }

    npy_intp *dims = PyArray_SHAPE(points);

    if (dims[1] != 2)
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - should be shape[1] == 2");
        Py_INCREF(Py_None);
        return Py_None;
    }

    size_t num = dims[0];

    const double *posi = PyArray_DATA(points);

    // We will use C functions directly for known projection types
    // Otherwise we will use python call

    struct ProjectionDef in_proj = 
    {
        .projection = input_proj,
        .forward = generic_forward_project,
        .reverse = generic_reverse_project,
    };

    struct ProjectionDef out_proj = 
    {
        .projection = output_proj,
        .forward = generic_forward_project,
        .reverse = generic_reverse_project,
    };
/*
    if (is_perspective(input_proj))
    {
        in_proj.projection = &(((struct PerspectiveProjectionObject *)input_proj)->proj);
        in_proj.forward = perspective_projection_project;
        in_proj.reverse = perspective_projection_reverse;
    }

    if (is_perspective(output_proj))
    {
        out_proj.projection = &(((struct PerspectiveProjectionObject *)output_proj)->proj);
        out_proj.forward = perspective_projection_project;
        out_proj.reverse = perspective_projection_reverse;
    }
    */

    PyArrayObject *output_points = (PyArrayObject *)PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
    if (output_points == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "can not allocate memory");
        Py_INCREF(Py_None);
        return Py_None;
    }
    double *poso = PyArray_DATA(output_points);

    fun(&self->mov, posi, poso, num, &in_proj, &out_proj);
    return (PyObject *)output_points;
}

static PyObject *Sphere_forward(PyObject *_self,
                                PyObject *args,
                                PyObject *kwds)
{
    return apply_action(_self, args, kwds, sphere_movement_apply_forward);
}

static PyObject *Sphere_reverse(PyObject *_self,
                                PyObject *args,
                                PyObject *kwds)
{
   return apply_action(_self, args, kwds, sphere_movement_apply_reverse);
}

static PyMethodDef _SphereMovements_methods[] = {
    {"apply_forward", (PyCFunction)Sphere_forward, METH_VARARGS | METH_KEYWORDS,
     "Apply forward rotation"},
    {"apply_reverse", (PyCFunction)Sphere_reverse, METH_VARARGS | METH_KEYWORDS,
     "Apply reverse rotation"},
    {NULL} /* Sentinel */
};

static PyTypeObject _SphereMovement = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = BASENAME ".SphereMovement",
    .tp_doc = PyDoc_STR("Sphere movement object"),
    .tp_basicsize = sizeof(struct SphereMovementObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = SphereMovements_init,
    .tp_methods = _SphereMovements_methods,
};

static PyModuleDef movementsModule = {
    PyModuleDef_HEAD_INIT,
    .m_name = BASENAME,
    .m_doc = "Movements module",
    .m_size = -1,
};


PyMODINIT_FUNC
PyInit_movements(void)
{
    import_array();
    PyObject *m = PyModule_Create(&movementsModule);
    if (m == NULL)
        return NULL;

    if (PyType_Ready(&_SphereMovement) < 0)
    {
        fprintf(stderr, "Bad type _SphereMovement\n");
        return NULL;
    }

    Py_INCREF(&_SphereMovement);
    if (PyModule_AddObject(m, "SphereMovement", (PyObject *)&_SphereMovement) < 0)
    {
        Py_DECREF(&_SphereMovement);
        Py_DECREF(m);
        fprintf(stderr, "Can not add SphereMovement\n");
        return NULL;
    }

    return m;
}
