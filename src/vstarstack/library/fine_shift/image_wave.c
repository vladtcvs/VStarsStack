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
#include <stdlib.h>
#include <string.h>

#define SQR(x) ((x)*(x))

struct ImageWaveObject
{
    PyObject_HEAD
    int Nw;         // Grid width
    int Nh;         // Grid height
    double w;       // Grid image width
    double h;       // Grid image height
    double sx;
    double sy;
    double stretch_penalty_k;

    double *array;  // Grid Nh x Nw x 2
    double *array_p;
    double *array_m;
    double *array_gradient;
};

/*
 * Set shift array at (x,y)
 */
static void set_array(double *array, int w, int h, int x, int y, int axis, double val)
{
    array[y*(w*2) + x*2 + axis] = val;
}

/*
 * Get shift array at (x,y)
 */
static double get_array(const double *array, int w, int h, int x, int y, int axis)
{
    if (x >= w)
        x = w-1;
    if (x < 0)
        x = 0;
    if (y >= h)
        y = h - 1;
    if (y < 0)
        y = 0;
    return array[y*(w*2) + x*2 + axis];
}

/*
 * Init shift array with specified (dx, dy)
 */
static void init_array(double *array, int w, int h, double dx, double dy)
{
    int xi, yi;
    for (yi = 0; yi < h; yi++)
    {
        for (xi = 0; xi < w; xi++)
        {
            set_array(array, w, h, xi, yi, 0, dx);
            set_array(array, w, h, xi, yi, 1, dy);
        }
    }
}

/*
 * Linear interpolation. x lies between points of f0 and f1
 */
static double bli(double f0, double f1, double x)
{
    return f0 * (1-x) + f1 * x;
}

/*
 * Cubic interpolation. x lies between points of f0 and f1
 */
static double bci(double fm1, double f0, double f1, double f2, double x)
{
    double a, b, c, d;
    d = f0;
    b = (fm1 + f1) / 2 - d;
    a = (f2 - 2*f1 - 2*b + d) / 6;
    c = f1 - b - d - a;
    return a*x*x*x + b*x*x + c*x + d;
}

/*
 * Bilinear interpolation. Use linear interpolation for x and y axes
 */
static void bilinear_interpolation(struct ImageWaveObject *self, const double *array,
                                    int xi, int yi, double dx, double dy,
                                    double *shift_x, double *shift_y)
{
    double left_top_x = get_array(array, self->Nw, self->Nh, xi, yi, 0);
    double right_top_x = get_array(array, self->Nw, self->Nh, xi+1, yi, 0);
    double left_bottom_x = get_array(array, self->Nw, self->Nh, xi, yi+1, 0);
    double right_bottom_x = get_array(array, self->Nw, self->Nh, xi+1, yi+1, 0);

    double left_top_y = get_array(array, self->Nw, self->Nh, xi, yi, 1);
    double right_top_y = get_array(array, self->Nw, self->Nh, xi+1, yi, 1);
    double left_bottom_y = get_array(array, self->Nw, self->Nh, xi, yi+1, 1);
    double right_bottom_y = get_array(array, self->Nw, self->Nh, xi+1, yi+1, 1);

    *shift_x = bli(bli(left_top_x, right_top_x, dx), bli(left_bottom_x, right_bottom_x, dx), dy);    
    *shift_y = bli(bli(left_top_y, right_top_y, dx), bli(left_bottom_y, right_bottom_y, dx), dy);    
}

/*
 * Bicubic interpolation. Use linear interpolation for x and y axes
 */
static void bicubic_interpolation(struct ImageWaveObject *self, const double *array,
                                  int xi, int yi, double dx, double dy,
                                  double *shift_x, double *shift_y)
{
    double x_m1m1 = get_array(array, self->Nw, self->Nh, xi-1, yi-1, 0);
    double x_0m1 = get_array(array, self->Nw, self->Nh, xi, yi-1, 0);
    double x_1m1 = get_array(array, self->Nw, self->Nh, xi+1, yi-1, 0);
    double x_2m1 = get_array(array, self->Nw, self->Nh, xi+2, yi-1, 0);

    double x_m10 = get_array(array, self->Nw, self->Nh, xi-1, yi, 0);
    double x_00 = get_array(array, self->Nw, self->Nh, xi, yi, 0);
    double x_10 = get_array(array, self->Nw, self->Nh, xi+1, yi, 0);
    double x_20 = get_array(array, self->Nw, self->Nh, xi+2, yi, 0);

    double x_m11 = get_array(array, self->Nw, self->Nh, xi-1, yi+1, 0);
    double x_01 = get_array(array, self->Nw, self->Nh, xi, yi+1, 0);
    double x_11 = get_array(array, self->Nw, self->Nh, xi+1, yi+1, 0);
    double x_21 = get_array(array, self->Nw, self->Nh, xi+2, yi+1, 0);
    
    double x_m12 = get_array(array, self->Nw, self->Nh, xi-1, yi+2, 0);
    double x_02 = get_array(array, self->Nw, self->Nh, xi, yi+2, 0);
    double x_12 = get_array(array, self->Nw, self->Nh, xi+1, yi+2, 0);
    double x_22 = get_array(array, self->Nw, self->Nh, xi+2, yi+2, 0);

    *shift_x = bci(bci(x_m1m1, x_0m1, x_1m1, x_2m1, dx),
                   bci(x_m10, x_00, x_10, x_20, dx),
                   bci(x_m11, x_01, x_11, x_21, dx),
                   bci(x_m12, x_02, x_12, x_22, dx),
                   dy);

    double y_m1m1 = get_array(array, self->Nw, self->Nh, xi-1, yi-1, 1);
    double y_0m1 = get_array(array, self->Nw, self->Nh, xi, yi-1, 1);
    double y_1m1 = get_array(array, self->Nw, self->Nh, xi+1, yi-1, 1);
    double y_2m1 = get_array(array, self->Nw, self->Nh, xi+2, yi-1, 1);

    double y_m10 = get_array(array, self->Nw, self->Nh, xi-1, yi, 1);
    double y_00 = get_array(array, self->Nw, self->Nh, xi, yi, 1);
    double y_10 = get_array(array, self->Nw, self->Nh, xi+1, yi, 1);
    double y_20 = get_array(array, self->Nw, self->Nh, xi+2, yi, 1);

    double y_m11 = get_array(array, self->Nw, self->Nh, xi-1, yi+1, 1);
    double y_01 = get_array(array, self->Nw, self->Nh, xi, yi+1, 1);
    double y_11 = get_array(array, self->Nw, self->Nh, xi+1, yi+1, 1);
    double y_21 = get_array(array, self->Nw, self->Nh, xi+2, yi+1, 1);
    
    double y_m12 = get_array(array, self->Nw, self->Nh, xi-1, yi+2, 1);
    double y_02 = get_array(array, self->Nw, self->Nh, xi, yi+2, 1);
    double y_12 = get_array(array, self->Nw, self->Nh, xi+1, yi+2, 1);
    double y_22 = get_array(array, self->Nw, self->Nh, xi+2, yi+2, 1);

    *shift_y = bci(bci(y_m1m1, y_0m1, y_1m1, y_2m1, dx),
                   bci(y_m10, y_00, y_10, y_20, dx),
                   bci(y_m11, y_01, y_11, y_21, dx),
                   bci(y_m12, y_02, y_12, y_22, dx),
                   dy);
}

/*
 * Interpolate values in shift array
 */
static void interpolate(struct ImageWaveObject *self, double *array,
                double x, double y, double *rx, double *ry)
{
    double sx = self->sx;
    double sy = self->sy;

    int xi = floor(x/sx);
    int yi = floor(y/sy);

    double dx = x/sx - xi;
    double dy = y/sy - yi;

    double shift_x, shift_y;

    bicubic_interpolation(self, array, xi, yi, dx, dy, &shift_x, &shift_y);

    *rx = x + shift_x;
    *ry = y + shift_y;
}

/*
 * Calculate penalty of shifts. We want to minimize it
 * Penalty contains of 2 parts:
 * 1. Penalty of points. It calculates from difference between actual
 * points shift and calculated from shift array
 * 2. Penalty of stretch. It calculates from difference between array shift values
 */
static double penalty(struct ImageWaveObject *self, double *array,
                        double *targets, double *points, size_t N)
{
    size_t i;
    double penalty_points = 0;
    for (i = 0; i < N; i++)
    {
        double x = points[i*2];
        double y = points[i*2+1];
        double tx = targets[i*2];
        double ty = targets[i*2+1];

        double sx, sy;
        interpolate(self, array, x, y, &sx, &sy);
        penalty_points += SQR(tx-sx) + SQR(ty-sy);
    }

    double penalty_stretch = 0;
    int xi, yi;
    for (yi = 0; yi < self->Nh-1; yi++)
    {
        for (xi = 0; xi < self->Nw-1; xi++)
        {
            double current_x = get_array(array, self->Nw, self->Nh, xi, yi, 0);
            double current_y = get_array(array, self->Nw, self->Nh, xi, yi, 1);
            double right_x = get_array(array, self->Nw, self->Nh, xi+1, yi, 0);
            double right_y = get_array(array, self->Nw, self->Nh, xi+1, yi, 1);
            double bottom_x = get_array(array, self->Nw, self->Nh, xi, yi+1, 0);
            double bottom_y = get_array(array, self->Nw, self->Nh, xi, yi+1, 1);

            penalty_stretch += SQR(current_x-right_x);
            penalty_stretch += SQR(current_y-right_y);
            
            penalty_stretch += SQR(current_x-bottom_x);
            penalty_stretch += SQR(current_y-bottom_y);
        }
    }
    return penalty_points * 1 + penalty_stretch * self->stretch_penalty_k;
}

/*
 * Calculate partial derivative of penalty by shift by axis <axis> at (x,y)
 */
static double partial(struct ImageWaveObject *self,
                        int yi, int xi, int axis,
                        double *targets, double *points, size_t N)
{
    double h = 1e-9;
    memcpy(self->array_p, self->array, self->Nw*self->Nh*2*sizeof(double));
    memcpy(self->array_m, self->array, self->Nw*self->Nh*2*sizeof(double));

    double val = get_array(self->array, self->Nw, self->Nh, xi, yi, axis);
    set_array(self->array_p, self->Nw, self->Nh, xi, yi, axis, val+h);
    set_array(self->array_m, self->Nw, self->Nh, xi, yi, axis, val-h);

    double penlaty_p = penalty(self, self->array_p, targets, points, N);
    double penlaty_m = penalty(self, self->array_m, targets, points, N);
    return (penlaty_p-penlaty_m)/(2*h);
}

/*
 * Step of gradient descent
 */
void approximate_step(struct ImageWaveObject *self, double dh,
                        double *targets, double *points, size_t N)
{
    int yi, xi;
    for (yi = 0; yi < self->Nh; yi++)
    {
        for (xi = 0; xi < self->Nw; xi++)
        {
            double gradient_x = partial(self, yi, xi, 0, targets, points, N);
            double gradient_y = partial(self, yi, xi, 1, targets, points, N);
            set_array(self->array_gradient, self->Nw, self->Nh, xi, yi, 0, gradient_x);
            set_array(self->array_gradient, self->Nw, self->Nh, xi, yi, 1, gradient_y);
        }
    }
    double maxv = 0;
    for (yi = 0; yi < self->Nh; yi++)
    {
        for (xi = 0; xi < self->Nw; xi++)
        {
            double gradient_x = get_array(self->array_gradient, self->Nw, self->Nh, xi, yi, 0);
            double gradient_y = get_array(self->array_gradient, self->Nw, self->Nh, xi, yi, 1);

            if (fabs(gradient_x) > maxv)
                maxv = fabs(gradient_x);
            if (fabs(gradient_y) > maxv)
                maxv = fabs(gradient_y);
        }
    }

    if (maxv > 1)
    {
        for (yi = 0; yi < self->Nh; yi++)
        {
            for (xi = 0; xi < self->Nw; xi++)
            {
                double gradient_x = get_array(self->array_gradient, self->Nw, self->Nh, xi, yi, 0);
                double gradient_y = get_array(self->array_gradient, self->Nw, self->Nh, xi, yi, 1);

                set_array(self->array_gradient, self->Nw, self->Nh, xi, yi, 0, gradient_x/maxv);
                set_array(self->array_gradient, self->Nw, self->Nh, xi, yi, 1, gradient_y/maxv);
            }
        }
    }

    for (yi = 0; yi < self->Nh; yi++)
    {
        for (xi = 0; xi < self->Nw; xi++)
        {
            double gradient_x = get_array(self->array_gradient, self->Nw, self->Nh, xi, yi, 0);
            double gradient_y = get_array(self->array_gradient, self->Nw, self->Nh, xi, yi, 1);
        
            double arr_x = get_array(self->array, self->Nw, self->Nh, xi, yi, 0);
            double arr_y = get_array(self->array, self->Nw, self->Nh, xi, yi, 1);

            set_array(self->array, self->Nw, self->Nh, xi, yi, 0, arr_x - gradient_x*dh);
            set_array(self->array, self->Nw, self->Nh, xi, yi, 1, arr_y - gradient_y*dh);
        }
    }
}

void approximate(struct ImageWaveObject *self, double dh, size_t Nsteps,
                        double *targets, double *points, size_t N)
{
    size_t i;
    if (N == 0)
        return;

    double dx = 0, dy = 0;
    for (i = 0; i < N; i++)
    {
        dx += targets[2*i] - points[2*i];
        dy += targets[2*i+1] - points[2*i+1];
    }
    dx /= N;
    dy /= N;

    init_array(self->array, self->Nw, self->Nh, dx, dy);

    for (i = 0; i < Nsteps; i++)
    {
        approximate_step(self, dh, targets, points, N);
    }
}

static int init(struct ImageWaveObject *self, double w, double h, double Nw, double Nh, double spk)
{
    self->w = w;
    self->h = h;
    self->Nw = Nw;
    self->Nh = Nh;
    self->stretch_penalty_k = spk;

    if (self->h <= 0 || self->w <= 0 || self->Nw < 2 || self->Nh < 2)
        return -1;

    self->sx = self->w / (self->Nw - 1);
    self->sy = self->h / (self->Nh - 1);

    self->array = calloc(self->Nw * self->Nh * 2, sizeof(double));
    if (!self->array)
    {
        return -1;
    }

    self->array_p = calloc(self->Nw * self->Nh * 2, sizeof(double));
    if (!self->array_p)
    {
        free(self->array);
        return -1;
    }

    self->array_m = calloc(self->Nw * self->Nh * 2, sizeof(double));
    if (!self->array_m)
    {
        free(self->array_p);
        free(self->array);
        return -1;
    }
    self->array_gradient = calloc(self->Nw * self->Nh * 2, sizeof(double));
    if (!self->array_gradient)
    {
        free(self->array_m);
        free(self->array_p);
        free(self->array);
        return -1;
    }
    return 0;
}

// arguments: w, h, Nw, Nh
static int ImageWave_init(PyObject *_self, PyObject *args, PyObject *kwds)
{
    double w, h;
    double spk;
    int Nw, Nh;
    struct ImageWaveObject *self = (struct ImageWaveObject *)_self;
    static char *kwlist[] = {"w", "h", "Nw", "Nh", "spk", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ddiid", kwlist,
                                     &w, &h, &Nw, &Nh, &spk))
        return -1;

    return init(self, w, h, Nw, Nh, spk);
}

static void ImageWave_finalize(PyObject *_self)
{
    PyObject *error_type, *error_value, *error_traceback;

    /* Save the current exception, if any. */
    PyErr_Fetch(&error_type, &error_value, &error_traceback);

    struct ImageWaveObject *self = (struct ImageWaveObject *)_self;
    if (self->array_gradient)
    {
        free(self->array_gradient);
        self->array_gradient = NULL;
    }
    if (self->array_m)
    {
        free(self->array_m);
        self->array_m = NULL;
    }
    if (self->array_p)
    {
        free(self->array_p);
        self->array_p = NULL;
    }
    if (self->array)
    {
        free(self->array);
        self->array = NULL;
    }

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
        return Py_None;

    double rx, ry;
    interpolate(self, self->array, x, y, &rx, &ry);
    return Py_BuildValue("(dd)", rx, ry);
}

static PyObject *ImageWave_approximate(PyObject *_self, PyObject *args, PyObject *kwds)
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
        return Py_None;
    }

    double *points_array = calloc(Npoints * 2, sizeof(double));
    if (!points_array)
    {
        free(targets_array);
        PyErr_SetString(PyExc_MemoryError, "insufficient memory");
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
    approximate(self, dh, Nsteps, targets_array, points_array, Npoints);
    free(targets_array);
    free(points_array);
    return Py_True;
}

static PyObject *ImageWave_data(PyObject *_self, PyObject *args, PyObject *kwds)
{
    struct ImageWaveObject *self = (struct ImageWaveObject *)_self;
    int xi, yi;
    PyObject *data = PyList_New(self->Nw*self->Nh*2);
    for (yi = 0; yi < self->Nh; yi++)
        for (xi = 0; xi < self->Nw; xi++)
        {
            double vx = get_array(self->array, self->Nw, self->Nh, xi, yi, 0);
            double vy = get_array(self->array, self->Nw, self->Nh, xi, yi, 1);

            PyList_SetItem(data, yi*self->Nw*2 + xi*2, PyFloat_FromDouble(vx));
            PyList_SetItem(data, yi*self->Nw*2 + xi*2 + 1, PyFloat_FromDouble(vy));
        }
    PyObject *result = Py_BuildValue("{s:i,s:i,s:d,s:d,s:d,s:O}",
                                        "Nw", self->Nw,
                                        "Nh", self->Nh,
                                        "w", self->w,
                                        "h", self->h,
                                        "spk", self->stretch_penalty_k,
                                        "data", data);
    return result;
}

static PyObject *ImageWave_fromdata(PyObject *_self, PyObject *args, PyObject *kwds);

static PyMethodDef ImageWave_methods[] = {
    {"interpolate", (PyCFunction)ImageWave_interpolate, METH_VARARGS | METH_KEYWORDS,
     "Apply shift grid to coordinates x,y"},
    {"approximate", (PyCFunction)ImageWave_approximate, METH_VARARGS | METH_KEYWORDS,
     "find grid values which gives the best fit for points -> targets"},
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

static PyObject *ImageWave_fromdata(PyObject *_self, PyObject *args, PyObject *kwds)
{
    int yi, xi;
    // _self == NULL
    PyObject *data;
    static char *kwlist[] = {"data", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &data))
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments");
        return Py_None;
    }

    double h = PyFloat_AsDouble(PyDict_GetItemString(data, "h"));
    double w = PyFloat_AsDouble(PyDict_GetItemString(data, "w"));
    long Nh  = PyLong_AsLong(PyDict_GetItemString(data, "Nh"));
    long Nw  = PyLong_AsLong(PyDict_GetItemString(data, "Nw"));
    double spk  = PyFloat_AsDouble(PyDict_GetItemString(data, "spk"));
    

    PyObject *argList = Py_BuildValue("ddiid", w, h, Nw, Nh, spk);
    PyObject *obj = PyObject_CallObject((PyObject *) &ImageWave, argList);

    Py_DECREF(argList);

    if (obj == NULL)
    {
        return Py_None;
    }

    struct ImageWaveObject *object = (struct ImageWaveObject *)obj;
    PyObject *values = PyDict_GetItemString(data, "data");

    if (PyList_Size(values) != Nw*Nh*2)
    {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_ValueError, "invalid values list len");
        return Py_None;
    }

    for (yi = 0; yi < Nh; yi++)
        for (xi = 0; xi < Nw; xi++)
        {
            int ind = (yi*Nw+xi)*2;
            double vx = PyFloat_AsDouble(PyList_GetItem(values, ind));
            double vy = PyFloat_AsDouble(PyList_GetItem(values, ind+1));

            set_array(object->array, Nw, Nh, xi, yi, 0, vx);
            set_array(object->array, Nw, Nh, xi, yi, 1, vy);
        }

    return obj;
}


static PyModuleDef image_waveModule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "vstarstack.library.fine_shift.image_wave",
    .m_doc = "Fine shift module - image_wave",
    .m_size = -1,
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

    return m;
}
