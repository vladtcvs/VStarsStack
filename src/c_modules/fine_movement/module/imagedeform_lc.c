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

#include "imagedeform_lc.h"
#include "imagedeform.h"
#include "imagegrid.h"

#include <image_deform_lc.h>

#include <numpy/ndarraytypes.h>
#include <numpy/ndarrayobject.h>

static int ImageDeformLC_init(PyObject *_self, PyObject *args, PyObject *kwds)
{
    int pixels;
    int image_w, image_h;
    struct ImageDeformLocalCorrelatorObject *self =
            (struct ImageDeformLocalCorrelatorObject *)_self;
    static char *kwlist[] = {"image_w", "image_h", "pixels", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iii", kwlist,
                                     &image_w, &image_h, &pixels))
        return -1;

    image_deform_lc_init(&self->correlator, image_w, image_h, pixels);
    return 0;
}

static void ImageDeformLC_dealloc(PyObject *_self)
{
    struct ImageDeformLocalCorrelatorObject *self =
            (struct ImageDeformLocalCorrelatorObject *)_self;
    image_deform_lc_finalize(&self->correlator);
}

static PyObject* ImageDeformLC_correlate(PyObject *_self, PyObject *args, PyObject *kwds)
{
    struct ImageDeformLocalCorrelatorObject *self =
            (struct ImageDeformLocalCorrelatorObject *)_self;

    int subpixels;
    int radius;
    double maximal_shift;
    PyObject *img, *ref_img;
    PyObject *pre_align, *ref_pre_align;
    static char *kwlist[] = {"img", "pre_align", "ref_img", "ref_pre_align",
                             "radius", "maximal_shift", "subpixels", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOidi", kwlist,
                                     &img, &pre_align, &ref_img, &ref_pre_align,
                                     &radius, &maximal_shift, &subpixels))
    {
        Py_INCREF(Py_None);
        return Py_None;
    }

    if (!PyObject_IsInstance(img, (PyObject *)&ImageGrid))
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - need ImageGrid");
        Py_INCREF(Py_None);
        return Py_None;
    }
    if (!PyObject_IsInstance(ref_img, (PyObject *)&ImageGrid))
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - need ImageGrid");
        Py_INCREF(Py_None);
        return Py_None;
    }
    if ((!Py_IsNone(pre_align)) && !PyObject_IsInstance(pre_align, (PyObject *)&ImageDeform))
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - need ImageDeform");
        Py_INCREF(Py_None);
        return Py_None;
    }
    if ((!Py_IsNone(ref_pre_align)) && !PyObject_IsInstance(ref_pre_align, (PyObject *)&ImageDeform))
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - need ImageDeform");
        Py_INCREF(Py_None);
        return Py_None;
    }

    struct ImageGridObject* _img = (struct ImageGridObject*)img;
    struct ImageGridObject* _ref_img = (struct ImageGridObject*)ref_img;

    struct ImageDeform *pre_align_img = NULL;
    struct ImageDeform *pre_align_ref_img = NULL;
    if (!Py_IsNone(pre_align))
        pre_align_img = &((struct ImageDeformObject *)pre_align)->deform;
    if (!Py_IsNone(ref_pre_align))
        pre_align_ref_img = &((struct ImageDeformObject *)ref_pre_align)->deform;

    image_deform_lc_find(&self->correlator, &_img->grid, pre_align_img,
                                            &_ref_img->grid, pre_align_ref_img,
                                            radius, maximal_shift, subpixels);

    const struct ImageDeform *deform = &self->correlator.array;
    PyObject *argList = Py_BuildValue("iiii", deform->image_w, deform->image_h,
                                              deform->grid_w, deform->grid_h);
    struct ImageDeformObject *deform_obj =
        (struct ImageDeformObject *)PyObject_CallObject((PyObject *)&ImageDeform, argList);
    Py_DECREF(argList);

    image_deform_set_shifts(&deform_obj->deform, deform->array);
    return (PyObject *)deform_obj;
}

static PyObject* ImageDeformLC_correlate_constant(PyObject *_self, PyObject *args, PyObject *kwds)
{
    struct ImageDeformLocalCorrelatorObject *self =
            (struct ImageDeformLocalCorrelatorObject *)_self;

    int subpixels;
    double maximal_shift;
    PyObject *img, *ref_img;
    PyObject *pre_align, *ref_pre_align;
    static char *kwlist[] = {"img", "pre_align", "ref_img", "ref_pre_align",
                             "maximal_shift", "subpixels", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOdi", kwlist,
                                     &img, &pre_align, &ref_img, &ref_pre_align,
                                     &maximal_shift, &subpixels))
    {
        Py_INCREF(Py_None);
        return Py_None;
    }

    if (!PyObject_IsInstance(img, (PyObject *)&ImageGrid))
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - need ImageGrid");
        Py_INCREF(Py_None);
        return Py_None;
    }
    if (!PyObject_IsInstance(ref_img, (PyObject *)&ImageGrid))
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - need ImageGrid");
        Py_INCREF(Py_None);
        return Py_None;
    }
    if ((!Py_IsNone(pre_align)) && !PyObject_IsInstance(pre_align, (PyObject *)&ImageDeform))
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - need ImageDeform");
        Py_INCREF(Py_None);
        return Py_None;
    }
    if ((!Py_IsNone(ref_pre_align)) && !PyObject_IsInstance(ref_pre_align, (PyObject *)&ImageDeform))
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - need ImageDeform");
        Py_INCREF(Py_None);
        return Py_None;
    }

    struct ImageGridObject* _img = (struct ImageGridObject*)img;
    struct ImageGridObject* _ref_img = (struct ImageGridObject*)ref_img;

    struct ImageDeform *pre_align_img = NULL;
    struct ImageDeform *pre_align_ref_img = NULL;
    if (!Py_IsNone(pre_align))
        pre_align_img = &((struct ImageDeformObject *)pre_align)->deform;
    if (!Py_IsNone(ref_pre_align))
        pre_align_ref_img = &((struct ImageDeformObject *)ref_pre_align)->deform;

    image_deform_lc_find_constant(&self->correlator, &_img->grid, pre_align_img,
                                  &_ref_img->grid, pre_align_ref_img,
                                  maximal_shift, subpixels);

    const struct ImageDeform *deform = &self->correlator.array;
    PyObject *argList = Py_BuildValue("iiii", deform->image_w, deform->image_h,
                                              deform->grid_w, deform->grid_h);
    struct ImageDeformObject *deform_obj =
        (struct ImageDeformObject *)PyObject_CallObject((PyObject *)&ImageDeform, argList);
    Py_DECREF(argList);

    image_deform_set_shifts(&deform_obj->deform, deform->array);
    return (PyObject *)deform_obj;
}


static PyMethodDef ImageDeformLC_methods[] = {
    {"find", (PyCFunction)ImageDeformLC_correlate, METH_VARARGS | METH_KEYWORDS,
     "Find image deformation to best local correlation"},
    {"find_constant", (PyCFunction)ImageDeformLC_correlate_constant, METH_VARARGS | METH_KEYWORDS,
     "Find image deformation to best local correlation"},
    {NULL} /* Sentinel */
};

PyTypeObject ImageDeformLC = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "vstarstack.library.fine_movement.module.ImageDeformLC",
    .tp_doc = PyDoc_STR("ImageDeform Local Correlator object"),
    .tp_basicsize = sizeof(struct ImageDeformLocalCorrelatorObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = ImageDeformLC_init,
    .tp_dealloc = ImageDeformLC_dealloc,
    .tp_methods = ImageDeformLC_methods,
};
