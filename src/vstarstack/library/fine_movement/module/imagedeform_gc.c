#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <image_deform_gc.h>
#include "imagedeform_gc.h"
#include "imagedeform.h"
#include <numpy/ndarraytypes.h>
#include <numpy/ndarrayobject.h>

static int ImageDeformGC_init(PyObject *_self, PyObject *args, PyObject *kwds)
{
    double spk;
    int image_w, image_h;
    int grid_w, grid_h;
    struct ImageDeformGlobalCorrelatorObject *self =
            (struct ImageDeformGlobalCorrelatorObject *)_self;
    static char *kwlist[] = {"image_w", "image_h", "grid_w", "grid_h", "spk", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiid", kwlist,
                                     &image_w, &image_h, &grid_w, &grid_h, &spk))
        return -1;

    image_deform_gc_init(&self->correlator, grid_w, grid_h, image_w, image_h, spk);
    return 0;
}

static void ImageDeformGC_finalize(PyObject *_self)
{
    struct ImageDeformGlobalCorrelatorObject *self =
            (struct ImageDeformGlobalCorrelatorObject *)_self;
    image_deform_gc_finalize(&self->correlator);
}

static PyObject* ImageDeformGC_correlate(PyObject *_self, PyObject *args, PyObject *kwds)
{
    double dh;
    int Nsteps;
    PyArrayObject *points;
    PyArrayObject *expected_points;

    struct ImageDeformGlobalCorrelatorObject *self =
            (struct ImageDeformGlobalCorrelatorObject *)_self;
    static char *kwlist[] = {"points", "expected_points", "dh", "Nsteps", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOd", kwlist,
                                     &points, &expected_points, &dh, &Nsteps))
    {
        Py_INCREF(Py_None);
        return Py_None;
    }

    const double *_points, *_expected_points;
    int npoints;

    if (PyArray_TYPE(points) != NPY_DOUBLE)
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - should be dtype == double");
        Py_INCREF(Py_None);
        return Py_None;
    }
    if (PyArray_TYPE(expected_points) != NPY_DOUBLE)
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - should be dtype == double");
        Py_INCREF(Py_None);
        return Py_None;
    }
    if (PyArray_NDIM(points) != 2)
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - should be dim == 2");
        Py_INCREF(Py_None);
        return Py_None;
    }
    if (PyArray_NDIM(expected_points) != 2)
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - should be dim == 2");
        Py_INCREF(Py_None);
        return Py_None;
    }

    npy_intp *dim_points = PyArray_SHAPE(points);
    npy_intp *dim_exp_points = PyArray_SHAPE(expected_points);
    
    if (dim_points[0] != dim_exp_points[0] || dim_points[1] != 2 || dim_exp_points[1] != 2)
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - bad dimensions");
        Py_INCREF(Py_None);
        return Py_None;
    }
    npoints = dim_points[0];
    _points = PyArray_DATA(points);
    _expected_points = PyArray_DATA(expected_points);

    const struct ImageDeform *deform = image_deform_gc_find(&self->correlator,
                                                            dh, Nsteps,
                                                            _points, _expected_points,
                                                            npoints);

    PyObject *argList = Py_BuildValue("iiii", deform->image_w, deform->image_h,
                                              deform->grid_w, deform->grid_h);
    struct ImageDeformObject *deform_obj =
        (struct ImageDeformObject *)PyObject_CallObject((PyObject *)&ImageDeform, argList);
    Py_DECREF(argList);

    image_deform_set_shifts(&deform_obj->deform, deform->array);
    return (PyObject *)deform_obj;
}

static PyMethodDef ImageDeformGC_methods[] = {
    {"find", (PyCFunction)ImageDeformGC_correlate, METH_VARARGS | METH_KEYWORDS,
     "Find image deformation to best global correlation"},
    {NULL} /* Sentinel */
};

PyTypeObject ImageDeformGC = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "vstarstack.library.fine_movement.ImageDeformGC",
    .tp_doc = PyDoc_STR("ImageDeform Global Correlator object"),
    .tp_basicsize = sizeof(struct ImageDeformGlobalCorrelatorObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = ImageDeformGC_init,
    .tp_finalize = ImageDeformGC_finalize,
    .tp_methods = ImageDeformGC_methods,
};
