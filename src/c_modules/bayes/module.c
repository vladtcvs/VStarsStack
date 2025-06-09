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

#define BASENAME "vstarstack.library.bayes.bayes"

static PyObject *posterior(PyObject *_self,
                           PyObject *args,
                           PyObject *kwds)
{
    PyObject *match_list;
    static char *kwlist[] = {"F", "f", "lambdas_d", "lambdas_v", "apriori", "apriori_params", "limits_low", "limits_high", "dl", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOOOOOf", kwlist, &match_list))
    {
        Py_INCREF(Py_None);
        return Py_None;
    }


}

static PyMethodDef bayes_methods[] = {
    {"bayes_estimation", (PyCFunction)bayes_estimation, METH_VARARGS | METH_KEYWORDS,
        "Find Bayes posterior mean"},

    {"bayes_map",       (PyCFunction)bayes_map, METH_VARARGS | METH_KEYWORDS,
        "Build Bayes MAP"},

    {"posterior",   (PyCFunction)posterior, METH_VARARGS | METH_KEYWORDS,
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

    return m;
}
