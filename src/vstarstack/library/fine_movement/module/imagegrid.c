#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <image_grid.h>
#include "imagegrid.h"
#include <numpy/ndarraytypes.h>
#include <numpy/ndarrayobject.h>

/**
 * \brief Init ImageGrid
 * \param _self ImageGridObject object
 * \param args arguments
 * \param kwads named arguments
 * \return 0 for OK
 */
static int ImageGrid_init(PyObject *_self, PyObject *args, PyObject *kwds)
{
    int image_w, image_h;
    struct ImageGridObject *self = (struct ImageGridObject *)_self;
    static char *kwlist[] = {"image_w", "image_h", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ii", kwlist,
                                     &image_w, &image_h))
        return -1;

    return image_grid_init(&self->grid, image_w, image_h);
}

/**
 * \brief Destroy ImageGrid
 * \param _self ImageGridObject object
 */
static void ImageGrid_finalize(PyObject *_self)
{
    PyObject *error_type, *error_value, *error_traceback;

    /* Save the current exception, if any. */
    PyErr_Fetch(&error_type, &error_value, &error_traceback);

    struct ImageGridObject *self = (struct ImageGridObject *)_self;
    image_grid_finalize(&self->grid);

    /* Restore the saved exception. */
    PyErr_Restore(error_type, error_value, error_traceback);
}

static PyObject *ImageGrid_fill(PyObject *_self,
                                PyObject *args,
                                PyObject *kwds)
{
    struct ImageGridObject *self = (struct ImageGridObject *)_self;
    static char *kwlist[] = {"image", NULL};
    PyArrayObject *image;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &image))
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
    if (PyArray_NDIM(image) != 2)
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - should be dim == 2");
        Py_INCREF(Py_None);
        return Py_None;
    }
    npy_intp *dims = PyArray_SHAPE(image);
    if (dims[0] == self->grid.h || dims[1] == self->grid.w)
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - image should be 2d array of correct size");
        Py_INCREF(Py_None);
        return Py_None;
    }
    image_grid_fill_pixels(&self->grid, PyArray_DATA(image));
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *ImageGrid_content(PyObject *_self,
                                   PyObject *args,
                                   PyObject *kwds)
{
    struct ImageGridObject *self = (struct ImageGridObject *)_self;
    npy_intp dims[2] = {self->grid.h, self->grid.w};
    PyArrayObject *image = (PyArrayObject *)PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
    double *data = PyArray_DATA(image);
    memcpy(data, self->grid.array, self->grid.h*self->grid.w*sizeof(double));
    return (PyObject *)image;
}

static PyMethodDef ImageGrid_methods[] = {
    {"fill", (PyCFunction)ImageGrid_fill, METH_VARARGS | METH_KEYWORDS,
     "Fill image grid from numpy array"},
    {"content", (PyCFunction)ImageGrid_content, METH_VARARGS | METH_KEYWORDS,
     "Return image grid content as numpy array"},
    {NULL} /* Sentinel */
};

PyTypeObject ImageGrid = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "vstarstack.library.fine_movement.ImageGrid",
    .tp_doc = PyDoc_STR("ImageGrid object"),
    .tp_basicsize = sizeof(struct ImageGridObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = ImageGrid_init,
    .tp_finalize = ImageGrid_finalize,
    .tp_methods = ImageGrid_methods,
};
