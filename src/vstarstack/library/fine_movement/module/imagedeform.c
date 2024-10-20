/*
 * Copyright (c) 2024 Vladislav Tsendrovskii
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

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL libdeform_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <image_deform.h>
#include "imagedeform.h"
#include "imagegrid.h"
#include <numpy/ndarraytypes.h>
#include <numpy/ndarrayobject.h>

/**
 * \brief Init ImageDeform
 * \param _self ImageDeformObject object
 * \param args arguments
 * \param kwads named arguments
 * \return 0 for OK
 */
static int ImageDeform_init(PyObject *_self, PyObject *args, PyObject *kwds)
{
    int image_w, image_h;
    int grid_w, grid_h;
    struct ImageDeformObject *self = (struct ImageDeformObject *)_self;
    static char *kwlist[] = {"image_w", "image_h", "grid_w", "grid_h", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiii", kwlist,
                                     &image_w, &image_h, &grid_w, &grid_h))
        return -1;

    return image_deform_init(&self->deform, grid_w, grid_h, image_w, image_h);
}

static void ImageDeform_finalize(PyObject *_self)
{
    PyObject *error_type, *error_value, *error_traceback;

    /* Save the current exception, if any. */
    PyErr_Fetch(&error_type, &error_value, &error_traceback);

    struct ImageDeformObject *self = (struct ImageDeformObject *)_self;
    image_deform_finalize(&self->deform);

    /* Restore the saved exception. */
    PyErr_Restore(error_type, error_value, error_traceback);
}

static PyObject *ImageDeform_fill(PyObject *_self,
                                  PyObject *args,
                                  PyObject *kwds)
{
    struct ImageDeformObject *self = (struct ImageDeformObject *)_self;
    static char *kwlist[] = {"shift_array", NULL};
    PyArrayObject *shift_array;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &shift_array))
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments");
        Py_INCREF(Py_None);
        return Py_None;
    }
    if (PyArray_TYPE(shift_array) != NPY_DOUBLE)
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - should be dtype == double");
        Py_INCREF(Py_None);
        return Py_None;
    }
    if (PyArray_NDIM(shift_array) != 3)
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - should be dim == 3");
        Py_INCREF(Py_None);
        return Py_None;
    }
    npy_intp *dims = PyArray_SHAPE(shift_array);
    if (dims[0] != self->deform.grid_h || dims[1] != self->deform.grid_w || dims[2] != 2)
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - image should be 3d array of correct size");
        Py_INCREF(Py_None);
        return Py_None;
    }
    image_deform_set_shifts(&self->deform, PyArray_DATA(shift_array));
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *ImageDeform_content(PyObject *_self,
                                     PyObject *args,
                                     PyObject *kwds)
{
    struct ImageDeformObject *self = (struct ImageDeformObject *)_self;
    npy_intp dims[3] = {self->deform.grid_h, self->deform.grid_w, 2};
    PyArrayObject *shift_array = (PyArrayObject *)PyArray_ZEROS(3, dims, NPY_DOUBLE, 0);
    if (shift_array == NULL)
    {
        Py_INCREF(Py_None);
        return Py_None;
    }
    double *data = PyArray_DATA(shift_array);
    int h = self->deform.grid_h;
    int w = self->deform.grid_w;
    memcpy(data, self->deform.array, h*w*2*sizeof(double));
    return (PyObject *)shift_array;
}

static PyObject *ImageDeform_apply_image(PyObject *_self,
                                         PyObject *args,
                                         PyObject *kwds)
{
    struct ImageDeformObject *self = (struct ImageDeformObject *)_self;
    static char *kwlist[] = {"image", "subpixels", NULL};
    PyObject *image;
    int subpixels;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Oi", kwlist, &image, &subpixels))
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments");
        Py_INCREF(Py_None);
        return Py_None;
    }
    if (!PyObject_IsInstance(image, (PyObject *)&ImageGrid))
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - need ImageGrid");
        Py_INCREF(Py_None);
        return Py_None;
    }
    struct ImageGridObject *in_img = (struct ImageGridObject *)image;

    PyObject *argList = Py_BuildValue("ii", in_img->grid.w*subpixels, in_img->grid.h*subpixels);
    struct ImageGridObject *out_img =
        (struct ImageGridObject *)PyObject_CallObject((PyObject *)&ImageGrid, argList);
    Py_DECREF(argList);

    image_deform_apply_image(&self->deform, &in_img->grid, &out_img->grid, subpixels);
    return (PyObject *)out_img;
}

static PyObject *ImageDeform_apply_point(PyObject *_self,
                                         PyObject *args,
                                         PyObject *kwds)
{
    struct ImageDeformObject *self = (struct ImageDeformObject *)_self;
    static char *kwlist[] = {"x", "y", NULL};
    double x, y;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "dd", kwlist, &x, &y))
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments");
        Py_INCREF(Py_None);
        return Py_None;
    }
    double sx, sy;
    image_deform_apply_point(&self->deform, x, y, &sx, &sy);
    PyObject *res = Py_BuildValue("dd", sx, sy);
    return res;
}

static PyMethodDef ImageDeform_methods[] = {
    {"fill", (PyCFunction)ImageDeform_fill, METH_VARARGS | METH_KEYWORDS,
     "Fill image deform from numpy array"},
    {"content", (PyCFunction)ImageDeform_content, METH_VARARGS | METH_KEYWORDS,
     "Return image deform content as numpy array"},
    {"apply_image", (PyCFunction)ImageDeform_apply_image, METH_VARARGS | METH_KEYWORDS,
     "Apply ImageDeform to ImageGrid"},
     {"apply_point", (PyCFunction)ImageDeform_apply_point, METH_VARARGS | METH_KEYWORDS,
     "Apply ImageDeform to point"},
    {NULL} /* Sentinel */
};

PyTypeObject ImageDeform = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "vstarstack.library.fine_movement.module.ImageDeform",
    .tp_doc = PyDoc_STR("ImageDeform object"),
    .tp_basicsize = sizeof(struct ImageDeformObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = ImageDeform_init,
    .tp_finalize = ImageDeform_finalize,
    .tp_methods = ImageDeform_methods,
};
