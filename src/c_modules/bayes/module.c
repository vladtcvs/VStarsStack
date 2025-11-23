/*
 * Copyright (c) 2025 Vladislav Tsendrovskii
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

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarraytypes.h>
#include <numpy/ndarrayobject.h>

#include <math.h>
#include <stdarg.h>

#include "bayes.h"

#define BASENAME "vstarstack.library.bayes.bayes"

struct apriori_params_s
{
    PyObject *apriori;
    PyObject *param_dict;
    double max_f;
};

enum BayesAprioriType_e
{
    APRIORI_UNKNOWN = 0,
    APRIORI_CALLABLE,
    APRIORI_UNIFORM,
    APRIORI_GAMMA,
};

struct BayesEstimatorObject
{
    PyObject_HEAD PyObject *apriori_callable_object;
    apriori_f apriori;
    double dl;
    enum BayesAprioriType_e apriori_type;
    int num_dim;
    struct bayes_posterior_ctx_s ctx;
};

double call_apriori(const double *f, int num_dim, const void *param)
{
    const struct apriori_params_s *params_s = param;

    PyObject *apriori = params_s->apriori;
    PyObject *param_dict = params_s->param_dict;

    npy_intp dims[1] = {num_dim};

    PyObject *f_ndarray = PyArray_SimpleNewFromData(
        1,          // ndim
        dims,       // dimensions
        NPY_DOUBLE, // dtype
        (void *)f   // pointer to your double data
    );

    if (!f_ndarray)
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create NumPy array");
        return 0;
    }

    PyObject *args = PyTuple_New(2);
    Py_INCREF(f_ndarray);  // Tuple steals a reference, so we must increment
    Py_INCREF(param_dict); // Tuple steals a reference, so we must increment
    PyTuple_SET_ITEM(args, 0, (PyObject *)f_ndarray);
    PyTuple_SET_ITEM(args, 1, (PyObject *)param_dict);

    PyObject *result = PyObject_CallObject(apriori, args);
    Py_DECREF(args); // Done with args tuple
    Py_DECREF(f_ndarray);

    if (!result)
    {
        PyErr_SetString(PyExc_RuntimeError, "Call to apriori function failed");
        return 0;
    }

    double apriori_val = PyFloat_AsDouble(result);
    Py_DECREF(result);

    if (PyErr_Occurred())
    {
        PyErr_SetString(PyExc_TypeError, "apriori function did not return a double");
        return 0;
    }

    return apriori_val;
}

double uniform_apriori(const double *f, int num_dim, const void *param)
{
    return 1;
}

static bool validate_shape(const PyArrayObject *array, int ndim, ...)
{
    if (PyArray_NDIM(array) != ndim)
        return false;

    va_list args;
    va_start(args, ndim);
    int i;
    for (i = 0; i < ndim; i++)
    {
        int expected_dim = va_arg(args, int);
        int actual_dim = PyArray_DIM(array, i);
        if (actual_dim != expected_dim)
        {
            va_end(args);
            return false;
        }
    }
    va_end(args);
    return true;
}

static PyObject *posterior(PyObject *_self,
                           PyObject *args,
                           PyObject *kwds)
{
    PyObject *F;
    PyObject *f;
    PyObject *lambdas_d, *lambdas_v, *lambdas_K;
    PyObject *apriori_params;
    PyObject *limits_low, *limits_high;

    struct BayesEstimatorObject *self = (struct BayesEstimatorObject *)_self;

    static char *kwlist[] = {"F", "f", "lambdas_d", "lambdas_v", "lambdas_K", "apriori_params", "limits_low", "limits_high", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOOOOO", kwlist,
                                     &F,
                                     &f,
                                     &lambdas_d, &lambdas_v, &lambdas_K,
                                     &apriori_params,
                                     &limits_low, &limits_high))
    {
        Py_INCREF(Py_None);
        return Py_None;
    }

    PyArrayObject *arr_F = (PyArrayObject *)PyArray_FROM_OTF(F, NPY_UINT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arr_f = (PyArrayObject *)PyArray_FROM_OTF(f, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arr_lambdas_d = (PyArrayObject *)PyArray_FROM_OTF(lambdas_d, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arr_lambdas_v = (PyArrayObject *)PyArray_FROM_OTF(lambdas_v, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arr_lambdas_K = (PyArrayObject *)PyArray_FROM_OTF(lambdas_K, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arr_limits_low = (PyArrayObject *)PyArray_FROM_OTF(limits_low, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arr_limits_high = (PyArrayObject *)PyArray_FROM_OTF(limits_high, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!arr_F || !arr_f || !arr_lambdas_d || !arr_lambdas_v || !arr_limits_low || !arr_limits_high)
    {
        PyErr_SetString(PyExc_TypeError, "Failed to convert inputs to NumPy arrays");
        goto fail;
    }

    // arr_F: [num_frames]
    if (PyArray_NDIM(arr_F) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "F must be a 1D array");
        goto fail;
    }
    int num_frames = PyArray_DIM(arr_F, 0);

    // arr_f: [num_frames]
    if (!validate_shape(arr_f, 1, self->num_dim))
    {
        PyErr_SetString(PyExc_ValueError, "f must be shape [num_dim]");
        goto fail;
    }

    // arr_lambdas_d: [num_dim]
    if (!validate_shape(arr_lambdas_d, 1, num_frames))
    {
        PyErr_SetString(PyExc_ValueError, "lambdas_d must be shape [num_frames]");
        goto fail;
    }

    // arr_lambdas_v: [num_frames]
    if (!validate_shape(arr_lambdas_v, 1, num_frames))
    {
        PyErr_SetString(PyExc_ValueError, "lambdas_v must be shape [num_frames]");
        goto fail;
    }

    // arr_lambdas_K: [num_frames, num_dim]
    if (!validate_shape(arr_lambdas_K, 2, num_frames, self->num_dim))
    {
        PyErr_SetString(PyExc_ValueError, "lambdas_K must be shape [num_frames, num_dim]");
        goto fail;
    }

    // arr_limits_low: [num_dim]
    if (!validate_shape(arr_limits_low, 1, self->num_dim))
    {
        PyErr_SetString(PyExc_ValueError, "limits_low must be shape [num_dim]");
        goto fail;
    }

    // arr_limits_high: [num_dim]
    if (!validate_shape(arr_limits_high, 1, self->num_dim))
    {
        PyErr_SetString(PyExc_ValueError, "limits_high must be shape [num_dim]");
        goto fail;
    }

    uint64_t *F_data = (uint64_t *)PyArray_DATA(arr_F);
    double *f_data = (double *)PyArray_DATA(arr_f);
    double *lambdas_d_data = (double *)PyArray_DATA(arr_lambdas_d);
    double *lambdas_v_data = (double *)PyArray_DATA(arr_lambdas_v);
    double *lambdas_K_data = (double *)PyArray_DATA(arr_lambdas_K);
    double *limits_low_data = (double *)PyArray_DATA(arr_limits_low);
    double *limits_high_data = (double *)PyArray_DATA(arr_limits_high);

    struct apriori_params_s params;
    switch (self->apriori_type)
    {
    case APRIORI_CALLABLE:
        params.apriori = self->apriori_callable_object;
        params.param_dict = apriori_params;
        break;
    case APRIORI_UNIFORM:
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "Invalid apriori type");
        goto fail;
    }

    double p = bayes_posterior(&self->ctx,
                                 num_frames,
                                 F_data,
                                 f_data,
                                 lambdas_d_data,
                                 lambdas_v_data,
                                 lambdas_K_data,
                                 self->apriori, &params,
                                 limits_low_data,
                                 limits_high_data,
                                 self->dl);

    Py_DECREF(arr_F);
    Py_DECREF(arr_f);
    Py_DECREF(arr_lambdas_d);
    Py_DECREF(arr_lambdas_v);
    Py_DECREF(arr_limits_low);
    Py_DECREF(arr_limits_high);
    return PyFloat_FromDouble(p);
fail:
    Py_XDECREF(arr_F);
    Py_XDECREF(arr_f);
    Py_XDECREF(arr_lambdas_d);
    Py_XDECREF(arr_lambdas_v);
    Py_XDECREF(arr_limits_low);
    Py_XDECREF(arr_limits_high);
    return NULL;
}

static PyObject *maxp(PyObject *_self,
                      PyObject *args,
                      PyObject *kwds)
{
    PyObject *F;
    PyObject *lambdas_d, *lambdas_v, *lambdas_K;
    PyObject *apriori_params;
    PyObject *limits_low, *limits_high;

    struct BayesEstimatorObject *self = (struct BayesEstimatorObject *)_self;

    static char *kwlist[] = {"F", "lambdas_d", "lambdas_v", "lambdas_K", "apriori_params", "limits_low", "limits_high", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOOOO", kwlist,
                                     &F,
                                     &lambdas_d, &lambdas_v, &lambdas_K,
                                     &apriori_params,
                                     &limits_low, &limits_high))
    {
        Py_INCREF(Py_None);
        return Py_None;
    }

    PyArrayObject *arr_F = (PyArrayObject *)PyArray_FROM_OTF(F, NPY_UINT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arr_lambdas_d = (PyArrayObject *)PyArray_FROM_OTF(lambdas_d, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arr_lambdas_v = (PyArrayObject *)PyArray_FROM_OTF(lambdas_v, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arr_lambdas_K = (PyArrayObject *)PyArray_FROM_OTF(lambdas_K, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arr_limits_low = (PyArrayObject *)PyArray_FROM_OTF(limits_low, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arr_limits_high = (PyArrayObject *)PyArray_FROM_OTF(limits_high, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!arr_F || !arr_lambdas_d || !arr_lambdas_v || !arr_limits_low || !arr_limits_high)
    {
        PyErr_SetString(PyExc_TypeError, "Failed to convert inputs to NumPy arrays");
        goto fail;
    }


    // arr_F: [num_frames]
    if (PyArray_NDIM(arr_F) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "F must be a 1D array");
        goto fail;
    }
    int num_frames = PyArray_DIM(arr_F, 0);

    // arr_lambdas_d: [num_frames]
    if (!validate_shape(arr_lambdas_d, 1, num_frames))
    {
        PyErr_SetString(PyExc_ValueError, "lambdas_d must be shape [num_frames]");
        goto fail;
    }

    // arr_lambdas_v: [num_frames]
    if (!validate_shape(arr_lambdas_v, 1, num_frames))
    {
        PyErr_SetString(PyExc_ValueError, "lambdas_v must be shape [num_frames]");
        goto fail;
    }

    // arr_lambdas_K: [num_frames, num_dim]
    if (!validate_shape(arr_lambdas_K, 2, num_frames, self->num_dim))
    {
        PyErr_SetString(PyExc_ValueError, "lambdas_K must be shape [num_frames, num_dim]");
        goto fail;
    }

    // arr_limits_low: [num_dim]
    if (!validate_shape(arr_limits_low, 1, self->num_dim))
    {
        PyErr_SetString(PyExc_ValueError, "limits_low must be shape [num_dim]");
        goto fail;
    }

    // arr_limits_high: [num_dim]
    if (!validate_shape(arr_limits_high, 1, self->num_dim))
    {
        PyErr_SetString(PyExc_ValueError, "limits_high must be shape [num_dim]");
        goto fail;
    }

    uint64_t *F_data = (uint64_t *)PyArray_DATA(arr_F);
    double *lambdas_d_data = (double *)PyArray_DATA(arr_lambdas_d);
    double *lambdas_v_data = (double *)PyArray_DATA(arr_lambdas_v);
    double *lambdas_K_data = (double *)PyArray_DATA(arr_lambdas_K);
    double *limits_low_data = (double *)PyArray_DATA(arr_limits_low);
    double *limits_high_data = (double *)PyArray_DATA(arr_limits_high);

    struct apriori_params_s params;
    switch (self->apriori_type)
    {
    case APRIORI_CALLABLE:
        params.apriori = self->apriori_callable_object;
        params.param_dict = apriori_params;
        break;
    case APRIORI_UNIFORM:
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "Invalid apriori type");
        goto fail;
    }

    npy_intp dims[1] = {self->num_dim};
    PyArrayObject *f_ndarray = (PyArrayObject *)PyArray_SimpleNew(
        1,         // ndim
        dims,      // dimensions
        NPY_DOUBLE // dtype
    );

    double *f = (double *)PyArray_DATA(f_ndarray);
    bayes_maxp(&self->ctx,
               num_frames,
               F_data,
               lambdas_d_data,
               lambdas_v_data,
               lambdas_K_data,
               self->apriori, &params,
               limits_low_data,
               limits_high_data,
               self->dl,
               f);
    Py_DECREF(arr_F);
    Py_DECREF(arr_lambdas_d);
    Py_DECREF(arr_lambdas_v);
    Py_DECREF(arr_limits_low);
    Py_DECREF(arr_limits_high);
    return (PyObject *)f_ndarray;
fail:
    Py_XDECREF(arr_F);
    Py_XDECREF(arr_lambdas_d);
    Py_XDECREF(arr_lambdas_v);
    Py_XDECREF(arr_limits_low);
    Py_XDECREF(arr_limits_high);
    return NULL;
}

static PyObject *estimate(PyObject *_self,
                          PyObject *args,
                          PyObject *kwds)
{
    PyObject *F;
    PyObject *lambdas_d, *lambdas_v, *lambdas_K;
    PyObject *apriori_params;
    PyObject *limits_low, *limits_high;
    double clip;

    struct BayesEstimatorObject *self = (struct BayesEstimatorObject *)_self;

    static char *kwlist[] = {"F", "lambdas_d", "lambdas_v", "lambdas_K", "apriori_params", "limits_low", "limits_high", "clip", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOOOOd", kwlist,
                                     &F,
                                     &lambdas_d, &lambdas_v, &lambdas_K,
                                     &apriori_params,
                                     &limits_low, &limits_high,
                                     &clip))
    {
        Py_INCREF(Py_None);
        return Py_None;
    }

    PyArrayObject *arr_F = (PyArrayObject *)PyArray_FROM_OTF(F, NPY_UINT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arr_lambdas_d = (PyArrayObject *)PyArray_FROM_OTF(lambdas_d, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arr_lambdas_v = (PyArrayObject *)PyArray_FROM_OTF(lambdas_v, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arr_lambdas_K = (PyArrayObject *)PyArray_FROM_OTF(lambdas_K, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arr_limits_low = (PyArrayObject *)PyArray_FROM_OTF(limits_low, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arr_limits_high = (PyArrayObject *)PyArray_FROM_OTF(limits_high, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!arr_F || !arr_lambdas_d || !arr_lambdas_v || !arr_limits_low || !arr_limits_high)
    {
        PyErr_SetString(PyExc_TypeError, "Failed to convert inputs to NumPy arrays");
        goto fail;
    }

    // arr_F: [num_frames]
    if (PyArray_NDIM(arr_F) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "F must be a 1D array");
        goto fail;
    }
    int num_frames = PyArray_DIM(arr_F, 0);

    // arr_lambdas_d: [num_frames]
    if (!validate_shape(arr_lambdas_d, 1, num_frames))
    {
        PyErr_SetString(PyExc_ValueError, "lambdas_d must be shape [num_frames]");
        goto fail;
    }

    // arr_lambdas_v: [num_frames]
    if (!validate_shape(arr_lambdas_v, 1, num_frames))
    {
        PyErr_SetString(PyExc_ValueError, "lambdas_v must be shape [num_frames]");
        goto fail;
    }

    // arr_lambdas_K: [num_frames, num_dim]
    if (!validate_shape(arr_lambdas_K, 2, num_frames, self->num_dim))
    {
        PyErr_SetString(PyExc_ValueError, "lambdas_K must be shape [num_frames, num_dim]");
        goto fail;
    }

    // arr_limits_low: [num_dim]
    if (!validate_shape(arr_limits_low, 1, self->num_dim))
    {
        PyErr_SetString(PyExc_ValueError, "limits_low must be shape [num_dim]");
        goto fail;
    }

    // arr_limits_high: [num_dim]
    if (!validate_shape(arr_limits_high, 1, self->num_dim))
    {
        PyErr_SetString(PyExc_ValueError, "limits_high must be shape [num_dim]");
        goto fail;
    }

    uint64_t *F_data = (uint64_t *)PyArray_DATA(arr_F);
    double *lambdas_d_data = (double *)PyArray_DATA(arr_lambdas_d);
    double *lambdas_v_data = (double *)PyArray_DATA(arr_lambdas_v);
    double *lambdas_K_data = (double *)PyArray_DATA(arr_lambdas_K);
    double *limits_low_data = (double *)PyArray_DATA(arr_limits_low);
    double *limits_high_data = (double *)PyArray_DATA(arr_limits_high);

    struct apriori_params_s params;
    switch (self->apriori_type)
    {
    case APRIORI_CALLABLE:
        params.apriori = self->apriori_callable_object;
        params.param_dict = apriori_params;
        break;
    case APRIORI_UNIFORM:
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "Invalid apriori type");
        goto fail;
    }

    npy_intp dims[1] = {self->num_dim};

    PyArrayObject *f_ndarray = (PyArrayObject *)PyArray_SimpleNew(
        1,         // ndim
        dims,      // dimensions
        NPY_DOUBLE // dtype
    );

    if (clip > 1)
        clip = 1;
    if (clip < 0)
        clip = 0;

    double *f = (double *)PyArray_DATA(f_ndarray);
    bayes_estimate(&self->ctx,
                   num_frames,
                   F_data,
                   lambdas_d_data,
                   lambdas_v_data,
                   lambdas_K_data,
                   self->apriori, &params,
                   limits_low_data,
                   limits_high_data,
                   self->dl, clip,
                   f);
    Py_DECREF(arr_F);
    Py_DECREF(arr_lambdas_d);
    Py_DECREF(arr_lambdas_v);
    Py_DECREF(arr_limits_low);
    Py_DECREF(arr_limits_high);
    return (PyObject *)f_ndarray;
fail:
    Py_XDECREF(arr_F);
    Py_XDECREF(arr_lambdas_d);
    Py_XDECREF(arr_lambdas_v);
    Py_XDECREF(arr_limits_low);
    Py_XDECREF(arr_limits_high);
    return NULL;
}

static int BayesEstimator_init(PyObject *_self, PyObject *args, PyObject *kwds)
{
    PyObject *apriori;
    double dl;
    int num_dim;

    static char *kwlist[] = {"apriori", "dl", "ndim", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Odi", kwlist, &apriori, &dl, &num_dim))
    {
        return -1;
    }
    struct BayesEstimatorObject *self = (struct BayesEstimatorObject *)_self;

    self->apriori_callable_object = NULL;
    self->apriori_type = APRIORI_UNKNOWN;
    self->dl = dl;
    self->num_dim = num_dim;

    // Callable apriori
    if (PyCallable_Check(apriori))
    {
        self->apriori = call_apriori;
        self->apriori_type = APRIORI_CALLABLE;
        Py_INCREF(apriori);
        self->apriori_callable_object = apriori;
    }

    // Uniform apriori
    if (PyUnicode_Check(apriori) && !strcmp(PyUnicode_AsUTF8(apriori), "uniform"))
    {
        self->apriori = uniform_apriori;
        self->apriori_type = APRIORI_UNIFORM;
    }

    if (self->apriori_type == APRIORI_UNKNOWN)
    {
        return -1;
    }

    if (!bayes_posterior_init(&self->ctx, self->num_dim))
    {
        PyErr_SetString(PyExc_ValueError, "initialization error");
        return -1;
    }

    return 0;
}

static void BayesEstimator_dealloc(PyObject *_self)
{
    struct BayesEstimatorObject *self =
        (struct BayesEstimatorObject *)_self;
    bayes_posterior_free(&self->ctx);
    if (self->apriori_callable_object)
    {
        Py_DECREF(self->apriori_callable_object);
        self->apriori_callable_object = NULL;
    }
}

static PyMethodDef BayesEstimator_methods[] = {
    {"posterior", (PyCFunction)posterior, METH_VARARGS | METH_KEYWORDS,
     "Build Bayes posterior"},

    {"MAP", (PyCFunction)maxp, METH_VARARGS | METH_KEYWORDS,
     "Bayes MAP value"},

    {"estimate", (PyCFunction)estimate, METH_VARARGS | METH_KEYWORDS,
     "Bayes estimate mean value"},

    {NULL} /* Sentinel */
};

static PyTypeObject bayesEstimator = {
    PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = BASENAME ".Estimator",
    .tp_doc = PyDoc_STR("Bayes estimator object"),
    .tp_basicsize = sizeof(struct BayesEstimatorObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = BayesEstimator_init,
    .tp_dealloc = BayesEstimator_dealloc,
    .tp_methods = BayesEstimator_methods,
};

static PyMethodDef bayes_methods[] = {
    {NULL} /* Sentinel */
};

static PyModuleDef bayesModule = {
    PyModuleDef_HEAD_INIT,
    .m_name = BASENAME,
    .m_doc = "Bayes estimation for Poisson distribution",
    .m_size = -1,
    .m_methods = bayes_methods,
};

PyMODINIT_FUNC
PyInit_bayes(void)
{
    import_array();
    if (PyType_Ready(&bayesEstimator) < 0)
        return NULL;

    PyObject *m = PyModule_Create(&bayesModule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&bayesEstimator);
    if (PyModule_AddObject(m, "BayesEstimator", (PyObject *)&bayesEstimator) < 0)
    {
        Py_DECREF(&bayesEstimator);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
