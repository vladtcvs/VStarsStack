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
#include <numpy/ndarraytypes.h>
#include <numpy/ndarrayobject.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "image_wave.h"

struct ImageWaveObject
{
    PyObject_HEAD
    struct ImageWave wave;
};

// arguments: w, h, Nw, Nh
static int ImageWave_init(PyObject *_self, PyObject *args, PyObject *kwds)
{
    int w, h;
    double spk;
    int Nw, Nh;
    struct ImageWaveObject *self = (struct ImageWaveObject *)_self;
    static char *kwlist[] = {"w", "h", "Nw", "Nh", "spk", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiid", kwlist,
                                     &w, &h, &Nw, &Nh, &spk))
        return -1;

    return image_wave_init(&self->wave, w, h, Nw, Nh, spk);
}

static void ImageWave_finalize(PyObject *_self)
{
    PyObject *error_type, *error_value, *error_traceback;

    /* Save the current exception, if any. */
    PyErr_Fetch(&error_type, &error_value, &error_traceback);

    struct ImageWaveObject *self = (struct ImageWaveObject *)_self;
    image_wave_finalize(&self->wave);

    /* Restore the saved exception. */
    PyErr_Restore(error_type, error_value, error_traceback);
}

static PyObject *ImageWave_interpolate(PyObject *_self, PyObject *args, PyObject *kwds)
{
    double x, y;
    struct ImageWaveObject *self = (struct ImageWaveObject *)_self;
    static char *kwlist[] = {"x", "y", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "dd", kwlist,
                                     &x, &y))
    {
        Py_INCREF(Py_None);
        return Py_None;
    }
    double rx, ry;
    image_wave_shift_interpolate(&self->wave, &self->wave.array, x, y, &rx, &ry);
    return Py_BuildValue("(dd)", rx, ry);
}

static PyObject *ImageWave_approximate_by_targets(PyObject *_self,
                                                  PyObject *args,
                                                  PyObject *kwds)
{
    size_t i;
    int Nsteps;
    double dh;
    PyObject *targets;
    PyObject *points;
    struct ImageWaveObject *self = (struct ImageWaveObject *)_self;
    static char *kwlist[] = {"targets", "points", "N", "dh", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOid", kwlist,
                                     &targets, &points, &Nsteps, &dh))
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments");
        Py_INCREF(Py_None);
        return Py_None;
    }

    if (!PyList_Check(targets))
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - targets MUST be list");
        return Py_None;
    }

    if (!PyList_Check(points))
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - points MUST be list");
        Py_INCREF(Py_None);
        return Py_None;
    }

    size_t Ntargets = PyList_Size(targets);
    size_t Npoints = PyList_Size(points);

    if (Ntargets != Npoints)
    {
        PyErr_SetString(PyExc_ValueError,
            "invalid function arguments - len(points) MUST be equal to len(targets)");
    }

    double *targets_array = calloc(Npoints * 2, sizeof(double));
    if (!targets_array)
    {
        PyErr_SetString(PyExc_MemoryError, "insufficient memory");
        Py_INCREF(Py_None);
        return Py_None;
    }

    double *points_array = calloc(Npoints * 2, sizeof(double));
    if (!points_array)
    {
        free(targets_array);
        PyErr_SetString(PyExc_MemoryError, "insufficient memory");
        Py_INCREF(Py_None);
        return Py_None;
    }

    for (i = 0; i < Npoints; i++)
    {
        PyObject *pnt = PyList_GetItem(points, i);
        PyObject *target = PyList_GetItem(targets, i);

        // TODO: add checks of tuples
        double pnt_x = PyFloat_AsDouble(PyTuple_GetItem(pnt, 0));
        double pnt_y = PyFloat_AsDouble(PyTuple_GetItem(pnt, 1));
        double target_x = PyFloat_AsDouble(PyTuple_GetItem(target, 0));
        double target_y = PyFloat_AsDouble(PyTuple_GetItem(target, 1));

        points_array[2*i] = pnt_x;
        points_array[2*i+1] = pnt_y;
        targets_array[2*i] = target_x;
        targets_array[2*i+1] = target_y;
    }
    image_wave_approximate_by_targets(&self->wave, dh, Nsteps, targets_array, points_array, Npoints);
    free(targets_array);
    free(points_array);
    Py_INCREF(Py_True);
    return Py_True;
}

static PyObject *ImageWave_approximate_by_correlation(PyObject *_self,
                                                      PyObject *args,
                                                      PyObject *kwds)
{
    int radius;
    double maximal_shift;

    PyArrayObject *image;
    PyArrayObject *ref_image;

    struct ImageWaveObject *self = (struct ImageWaveObject *)_self;
    static char *kwlist[] = {"image", "reference_image", "radius", "maximal_shift", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOid", kwlist,
                                     &image, &ref_image, &radius, &maximal_shift))
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments");
        Py_INCREF(Py_None);
        return Py_None;
    }

    if (PyArray_TYPE(image) != NPY_DOUBLE)
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - should be dtype == double");
        Py_INCREF(Py_None);
        return Py_None;
    }

    if (PyArray_TYPE(ref_image) != NPY_DOUBLE)
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - should be dtype == double");
        Py_INCREF(Py_None);
        return Py_None;
    }

    npy_intp *dims = PyArray_SHAPE(image);
    struct ImageWaveGrid img = {
        .array = PyArray_DATA(image),
        .naxis = 1,
        .w = dims[1],
        .h = dims[0],
    };

    npy_intp *ref_dims = PyArray_SHAPE(ref_image);
    struct ImageWaveGrid ref_img = {
        .array = PyArray_DATA(ref_image),
        .naxis = 1,
        .w = ref_dims[1],
        .h = ref_dims[0],
    };

    // TODO: implement

    return PyFloat_FromDouble(0);
}

static PyObject *ImageCorrelation(PyObject *self,
                             PyObject *args,
                             PyObject *kwds)
{
    PyArrayObject *image1;
    PyArrayObject *image2;

    static char *kwlist[] = {"image1", "image2", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", kwlist, &image1, &image2))
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments");
        Py_INCREF(Py_None);
        return Py_None;
    }

    if (PyArray_TYPE(image1) != NPY_DOUBLE)
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - should be dtype == double");
        Py_INCREF(Py_None);
        return Py_None;
    }

    if (PyArray_TYPE(image2) != NPY_DOUBLE)
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - should be dtype == double");
        Py_INCREF(Py_None);
        return Py_None;
    }


    npy_intp *dims1 = PyArray_SHAPE(image1);
    struct ImageWaveGrid img1 = {
        .array = PyArray_DATA(image1),
        .naxis = 1,
        .w = dims1[1],
        .h = dims1[0],
    };

    npy_intp *dims2 = PyArray_SHAPE(image2);
    struct ImageWaveGrid img2 = {
        .array = PyArray_DATA(image2),
        .naxis = 1,
        .w = dims2[1],
        .h = dims2[0],
    };

    if (img1.w != img2.w || img1.h != img2.h)
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - should be same shape");
        Py_INCREF(Py_None);
        return Py_None;
    }

    double correlation = image_wave_correlation(&img1, &img2);
    return PyFloat_FromDouble(correlation);
}

static PyObject *ImageWave_apply_shift(PyObject *_self,
                                       PyObject *args,
                                       PyObject *kwds)
{
    PyArrayObject *image;
    struct ImageWaveObject *self = (struct ImageWaveObject *)_self;
    static char *kwlist[] = {"image", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist,
                                     &image))
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments");
        Py_INCREF(Py_None);
        return Py_None;
    }

    if (PyArray_NDIM(image) != 2)
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - should be dim == 2");
        Py_INCREF(Py_None);
        return Py_None;
    }

    if (PyArray_TYPE(image) != NPY_DOUBLE)
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - should be dtype == double");
        Py_INCREF(Py_None);
        return Py_None;
    }

    npy_intp *dims = PyArray_SHAPE(image);
    struct ImageWaveGrid img = {
        .array = PyArray_DATA(image),
        .naxis = 1,
        .w = dims[1],
        .h = dims[0],
    };

    PyArrayObject *output_image = (PyArrayObject *)PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
    struct ImageWaveGrid out = {
        .array = PyArray_DATA(output_image),
        .naxis = 1,
        .w = dims[1],
        .h = dims[0],
    };

    image_wave_shift_image(&self->wave, &self->wave.array, &img, &out);
    Py_INCREF(output_image);
    return (PyObject *)output_image;
}

static PyObject *ImageWave_data(PyObject *_self, PyObject *args, PyObject *kwds)
{
    struct ImageWaveObject *self = (struct ImageWaveObject *)_self;
    int xi, yi;
    PyObject *data = PyList_New(self->wave.array.w * self->wave.array.h * 2);
    for (yi = 0; yi < self->wave.array.h; yi++)
        for (xi = 0; xi < self->wave.array.w; xi++)
        {
            double vx = image_wave_get_array(&self->wave.array,
                                             xi, yi, 0);
            double vy = image_wave_get_array(&self->wave.array,
                                             xi, yi, 1);

            PyList_SetItem(data, yi*self->wave.array.w*2 + xi*2, PyFloat_FromDouble(vx));
            PyList_SetItem(data, yi*self->wave.array.w*2 + xi*2 + 1, PyFloat_FromDouble(vy));
        }
    PyObject *result = Py_BuildValue("{s:i,s:i,s:i,s:i,s:d,s:O}",
                                        "Nw", self->wave.array.w,
                                        "Nh", self->wave.array.h,
                                        "w", self->wave.w,
                                        "h", self->wave.h,
                                        "spk", self->wave.stretch_penalty_k,
                                        "data", data);
    return result;
}

static PyObject *ImageWave_fromdata(PyObject *_self, PyObject *args, PyObject *kwds);

static PyMethodDef ImageWave_methods[] = {
    {"interpolate", (PyCFunction)ImageWave_interpolate, METH_VARARGS | METH_KEYWORDS,
     "Apply shift grid to coordinates x,y"},

    {"approximate_by_targets", (PyCFunction)ImageWave_approximate_by_targets, METH_VARARGS | METH_KEYWORDS,
     "find grid values which gives the best fit for points -> targets"},

    {"approximate_by_correlation", (PyCFunction)ImageWave_approximate_by_correlation, METH_VARARGS | METH_KEYWORDS,
     "find grid values which gives the best correlation between image1 and image2"},

    {"apply_shift", (PyCFunction)ImageWave_apply_shift, METH_VARARGS | METH_KEYWORDS,
     "apply shift grid to image"},

    {"data", (PyCFunction)ImageWave_data, METH_VARARGS | METH_KEYWORDS,
     "data of ImageWave"},

    {"from_data", (PyCFunction)ImageWave_fromdata, METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     "generate ImageWave from data"},

    {NULL} /* Sentinel */
};

static PyTypeObject ImageWave = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "vstarstack.library.fine_shift.image_wave.ImageWave",
    .tp_doc = PyDoc_STR("ImageWave object"),
    .tp_basicsize = sizeof(struct ImageWaveObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = ImageWave_init,
    .tp_finalize = ImageWave_finalize,
    .tp_methods = ImageWave_methods,
};

static PyMethodDef methods[] = {
    {"image_correlation", (PyCFunction)ImageCorrelation, METH_VARARGS | METH_KEYWORDS, "find correlation between images"},
    {NULL, NULL, 0, NULL},
};

static PyObject *ImageWave_fromdata(PyObject *_self, PyObject *args, PyObject *kwds)
{
    int yi, xi;
    // _self == NULL
    PyObject *data;
    static char *kwlist[] = {"data", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &data))
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments");
        Py_INCREF(Py_None);
        return Py_None;
    }

    int h = PyFloat_AsDouble(PyDict_GetItemString(data, "h"));
    int w = PyFloat_AsDouble(PyDict_GetItemString(data, "w"));
    long Nh  = PyLong_AsLong(PyDict_GetItemString(data, "Nh"));
    long Nw  = PyLong_AsLong(PyDict_GetItemString(data, "Nw"));
    double spk  = PyFloat_AsDouble(PyDict_GetItemString(data, "spk"));
    

    PyObject *argList = Py_BuildValue("iiiid", w, h, Nw, Nh, spk);
    PyObject *obj = PyObject_CallObject((PyObject *) &ImageWave, argList);

    Py_DECREF(argList);

    if (obj == NULL)
    {
        Py_INCREF(Py_None);
        return Py_None;
    }

    struct ImageWaveObject *object = (struct ImageWaveObject *)obj;
    PyObject *values = PyDict_GetItemString(data, "data");

    if (PyList_Size(values) != Nw*Nh*2)
    {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_ValueError, "invalid values list len");
        Py_INCREF(Py_None);
        return Py_None;
    }

    for (yi = 0; yi < Nh; yi++)
        for (xi = 0; xi < Nw; xi++)
        {
            int ind = (yi*Nw+xi)*2;
            double vx = PyFloat_AsDouble(PyList_GetItem(values, ind));
            double vy = PyFloat_AsDouble(PyList_GetItem(values, ind+1));

            image_wave_set_array(&object->wave.array, xi, yi, 0, vx);
            image_wave_set_array(&object->wave.array, xi, yi, 1, vy);
        }

    Py_INCREF(obj);
    return obj;
}


static PyModuleDef image_waveModule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "vstarstack.library.fine_shift.image_wave",
    .m_doc = "Fine shift module - image_wave",
    .m_size = -1,
    .m_methods = methods,
};

PyMODINIT_FUNC
PyInit_image_wave(void)
{
    PyObject *m;
    if (PyType_Ready(&ImageWave) < 0)
        return NULL;

    m = PyModule_Create(&image_waveModule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&ImageWave);
    if (PyModule_AddObject(m, "ImageWave", (PyObject *)&ImageWave) < 0)
    {
        Py_DECREF(&ImageWave);
        Py_DECREF(m);
        return NULL;
    }

    import_array();
    return m;
}
