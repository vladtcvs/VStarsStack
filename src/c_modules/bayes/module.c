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

#include "bayes.h"

#define BASENAME "vstarstack.library.bayes.bayes"

struct default_apriori_params_s
{
    PyObject *apriori;
    PyObject *param_dict;
};

float call_apriori(const float *f, int num_dim, void *param)
{
    struct default_apriori_params_s *params_s = param;
    
    PyObject *apriori = params_s->apriori;
    PyObject *param_dict = params_s->param_dict;
 
    npy_intp dims[1] = { num_dim };

    PyObject *f_ndarray = PyArray_SimpleNewFromData(
        1,               // ndim
        dims,            // dimensions
        NPY_FLOAT32,     // dtype
        (void *)f        // pointer to your float data
    );

    if (!array) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create NumPy array");
        return 0;
    }

    PyObject *args = PyTuple_New(2);
    Py_INCREF(f_ndarray);  // Tuple steals a reference, so we must increment
    Py_INCREF(param_dict); // Tuple steals a reference, so we must increment
    PyTuple_SET_ITEM(args, 0, (PyObject *)f_ndarray);
    PyTuple_SET_ITEM(args, 1, (PyObject *)param_dict);
    
    PyObject *result = PyObject_CallObject(apriori, args);
    Py_DECREF(args);  // Done with args tuple
    Py_DECREF(f_ndarray);

    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Call to apriori function failed");
        return 0;
    }
    
    double apriori_val = PyFloat_AsDouble(result);
    Py_DECREF(result);
    
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "apriori function did not return a float");
        return 0;
    }

    return apriori_val;
}

static PyObject *posterior(PyObject *_self,
                           PyObject *args,
                           PyObject *kwds)
{
    PyObject *F;
    PyObject *f;
    PyObject *lambdas_d, *lambdas_v;
    PyObject *apriori, *apriori_params;
    PyObject *limits_low, *limits_high;
    float dl;

    static char *kwlist[] = {"F", "f", "lambdas_d", "lambdas_v", "apriori", "apriori_params", "limits_low", "limits_high", "dl", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOOOOOf", kwlist,
                                     &F,
                                     &f,
                                     &lambdas_d, &lambdas_v,
                                     &apriori, &apriori_params,
                                     &limits_low, &limits_high,
                                     &dl))
    {
        Py_INCREF(Py_None);
        return Py_None;
    }

    PyArrayObject *arr_F = (PyArrayObject *)PyArray_FROM_OTF(F, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arr_f = (PyArrayObject *)PyArray_FROM_OTF(f, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arr_lambdas_d = (PyArrayObject *)PyArray_FROM_OTF(lambdas_d, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arr_lambdas_v = (PyArrayObject *)PyArray_FROM_OTF(lambdas_v, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arr_limits_low = (PyArrayObject *)PyArray_FROM_OTF(limits_low, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arr_limits_high = (PyArrayObject *)PyArray_FROM_OTF(limits_high, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

    if (!arr_F || !arr_f || !arr_lambdas_d || !arr_lambdas_v || !arr_limits_low || !arr_limits_high)
    {
        PyErr_SetString(PyExc_TypeError, "Failed to convert inputs to NumPy arrays");
        goto fail;
    }

    // arr_f: [num_frames]
    if (PyArray_NDIM(arr_F) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "F must be a 1D array");
        goto fail;
    }
    npy_intp num_frames = PyArray_DIM(arr_F, 0);

    // arr_f: [num_dim]
    if (PyArray_NDIM(arr_f) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "f must be a 1D array");
        goto fail;
    }
    npy_intp num_dim = PyArray_DIM(arr_f, 0);

    // arr_lambdas_d: [num_dim]
    if (PyArray_NDIM(arr_lambdas_d) != 1 ||
        PyArray_DIM(arr_lambdas_v, 0) != num_frames)
    {
        PyErr_SetString(PyExc_ValueError, "lambdas_d must be shape [num_frames]");
        goto fail;
    }

    // arr_lambdas_v: [num_frames, num_dim]
    if (PyArray_NDIM(arr_lambdas_v) != 2 ||
        PyArray_DIM(arr_lambdas_v, 0) != num_frames ||
        PyArray_DIM(arr_lambdas_v, 1) != num_dim)
    {
        PyErr_SetString(PyExc_ValueError, "lambdas_v must be shape [num_frames, num_dim]");
        goto fail;
    }

    // arr_limits_low: [num_dim]
    if (PyArray_NDIM(arr_limits_low) != 1 ||
        PyArray_DIM(arr_limits_low, 0) != num_frames)
    {
        PyErr_SetString(PyExc_ValueError, "limits_low must be shape [num_dim]");
        goto fail;
    }

    // arr_limits_high: [num_dim]
    if (PyArray_NDIM(arr_limits_high) != 1 ||
        PyArray_DIM(arr_limits_high, 0) != num_dim)
    {
        PyErr_SetString(PyExc_ValueError, "limits_high must be shape [num_dim]");
        goto fail;
    }

    if (!(PyCallable_Check(apriori) || PyUnicode_Check(apriori))) {
        PyErr_SetString(PyExc_TypeError, "apriori must be a string or callable");
        goto fail;
    }

    int *F_data = (int *)PyArray_DATA(arr_F);
    float *f_data = (float *)PyArray_DATA(arr_f);
    float *lambdas_d_data = (float *)PyArray_DATA(arr_lambdas_d);
    float *lambdas_v_data = (float *)PyArray_DATA(arr_lambdas_v);
    float *limits_low_data = (float *)PyArray_DATA(arr_limits_low);
    float *limits_high_data = (float *)PyArray_DATA(arr_limits_high);

    struct 

    struct bayes_posterior_ctx_s ctx;
    if (!bayes_posterior_init(&ctx, num_dim))
    {
        PyErr_SetString(PyExc_ValueError, "initialization error");
        goto fail;
    }

    apriori_f apriori_fun = NULL;
    void *apriori_params = NULL;

    // Callable apriori
    struct default_apriori_params_s params;
    if (PyCallable_Check(apriori)) {    
        params.apriori = apriori;
        params.param_dict = apriori_params;
        apriori_fun = call_apriori;
        apriori_params = &params;
    }

    // Uniform apriori
    float max_f;
    if (PyUnicode_Check(apriori) && !strcmp(PyUnicode_AsUTF8(apriori), "uniform")) {
        apriori_fun = uniform_apriori;
        PyObject *val = PyDict_GetItemString(apriori_params, "f_max");  // no new ref
        if (!val) {
            PyErr_SetString(PyExc_ValueError, "apriori params must have 'f_max' key");
            goto fail;
        }
        max_f = PyFloat_AsDouble(val);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "Value for 'f_max' is not a float");
            goto fail;
        }
        apriori_params = &max_f;
    }


    if (apriori_fun == NULL) {
        PyErr_SetString(PyExc_TypeError, "apriori function not setted");
        goto fail;
    }
    float p = bayes_posterior(&ctx,
                              num_frames,
                              F_data,
                              f_data,
                              lambdas_d_data, lambdas_v_data,
                              apriori_fun, &apriori_params,
                              limits_low_data, limits_high_data, dl);
    }
    bayes_posterior_free(&ctx);
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

static PyMethodDef bayes_methods[] = {
    {"bayes_estimation", (PyCFunction)estimation, METH_VARARGS | METH_KEYWORDS,
     "Find Bayes posterior mean"},

    {"bayes_map", (PyCFunction)map, METH_VARARGS | METH_KEYWORDS,
     "Build Bayes MAP"},

    {"posterior", (PyCFunction)posterior, METH_VARARGS | METH_KEYWORDS,
     "Build Bayes posterior"},

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
    PyObject *m = PyModule_Create(&bayesModule);
    if (m == NULL)
        return NULL;
    import_array();
    return m;
}
