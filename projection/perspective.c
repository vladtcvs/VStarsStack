#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#include <math.h>
#include <stdio.h>

struct ProjectionObject
{
    PyObject_HEAD
    double H;       // height of image in mm
    double W;       // width of image in mm
    double F;       // focal length in mm
    int h;          // height of image in pixels
    int w;          // width of image in pixels
    double kx;      // transformation from pixels to mm coefficient
    double ky;      // transformation from pixels to mm coefficient
};

// arguments: W, H, F, w, h
static int Projection_init(PyObject *_self, PyObject *args, PyObject *kwds)
{
    struct ProjectionObject *self = (struct ProjectionObject *)_self;
    static char *kwlist[] = {"W", "H", "F", "w", "h", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "dddii", kwlist,
                                     &self->W, &self->H, &self->F, &self->w, &self->h))
        return -1;
    if (self->h <= 0 || self->w <= 0 || self->W <= 0 || self->H <= 0 || self->F <= 0)
        return -1;

    self->kx = self->W / self->w;
    self->ky = self->H / self->h;
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
    double X = (self->w / 2 - x) * self->kx;
    double Y = (self->h / 2 - y) * self->ky;
    double lon = atan(X / self->F);
    double lat = atan(Y * cos(lon) / self->F);

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

    double X = self->F * tan(lon);
    double Y = self->F * tan(lat) / cos(lon);
    double x = (self->w / 2 - X / self->kx);
    double y = (self->h / 2 - Y / self->ky);
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
    .tp_name = "projection.perspective.Projection",
    .tp_doc = PyDoc_STR("Perspective projection object"),
    .tp_basicsize = sizeof(struct ProjectionObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = Projection_init,
    .tp_methods = Projection_methods,
};

static PyModuleDef perspectiveModule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "projection.perspective",
    .m_doc = "Perspective projection module",
    .m_size = -1,
};

PyMODINIT_FUNC
PyInit_perspective(void)
{
    PyObject *m;
    if (PyType_Ready(&Projection) < 0)
        return NULL;

    m = PyModule_Create(&perspectiveModule);
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
