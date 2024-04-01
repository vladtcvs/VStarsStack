#include "imagedeform_lc.h"
#include "imagedeform.h"
#include "imagegrid.h"

#include <image_deform_lc.h>

static int ImageDeformLC_init(PyObject *_self, PyObject *args, PyObject *kwds)
{
    int pixels;
    int image_w, image_h;
    struct ImageDeformLocalCorrelatorObject *self =
            (struct ImageDeformLocalCorrelatorObject *)_self;
    static char *kwlist[] = {"image_w", "image_h", "pixels", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiid", kwlist,
                                     &image_w, &image_h, &pixels))
        return -1;

    image_deform_lc_init(&self->correlator, image_w, image_h, pixels);
    return 0;
}

static void ImageDeformLC_finalize(PyObject *_self)
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
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - need ImageDeform");
        Py_INCREF(Py_None);
        return Py_None;
    }
    if (!PyObject_IsInstance(ref_img, (PyObject *)&ImageGrid))
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - need ImageDeform");
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
    struct ImageDeformObject *_pre_align = (struct ImageDeformObject *)pre_align;
    struct ImageDeformObject *_ref_pre_align = (struct ImageDeformObject *)ref_pre_align;

    image_deform_lc_find(&self->correlator, &_img->grid, &_pre_align->deform,
                                            &_ref_img->grid, &_ref_pre_align->deform,
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

static PyMethodDef ImageDeformLC_methods[] = {
    {"find", (PyCFunction)ImageDeformLC_correlate, METH_VARARGS | METH_KEYWORDS,
     "Find image deformation to best local correlation"},
    {NULL} /* Sentinel */
};

PyTypeObject ImageDeformLC = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "vstarstack.library.fine_movement.ImageDeformLC",
    .tp_doc = PyDoc_STR("ImageDeform Local Correlator object"),
    .tp_basicsize = sizeof(struct ImageDeformLocalCorrelatorObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = ImageDeformLC_init,
    .tp_finalize = ImageDeformLC_finalize,
    .tp_methods = ImageDeformLC_methods,
};
