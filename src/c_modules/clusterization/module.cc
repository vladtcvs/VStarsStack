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


#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include <vector>

#include "clusters.hpp"

#define BASENAME "vstarstack.library.clusters.clusterization"


static PyObject *build_clusters_from_matches(PyObject *_self,
                                             PyObject *args,
                                             PyObject *kwds)
{
    PyObject *match_list;
    char match_list_key[] = "match_list";
    static char *kwlist[] = {match_list_key, NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &match_list))
    {
        Py_INCREF(Py_None);
        return Py_None;
    }

    if (!PyList_Check(match_list))
    {
        PyErr_SetString(PyExc_ValueError, "invalid function arguments - expect list");
        Py_INCREF(Py_None);
        return Py_None;
    }

    auto match_list_len = PyList_GET_SIZE(match_list);
    std::vector<match_s> matches;
    matches.resize(match_list_len);
    for (auto index = 0; index < match_list_len; index++)
    {
        PyObject *match_item = PyList_GET_ITEM(match_list, index);
        if (!PyTuple_Check(match_item))
        {
            PyErr_SetString(PyExc_ValueError, "invalid function arguments - expect tuples in list");
            Py_INCREF(Py_None);
            return Py_None;
        }

        if (PyTuple_GET_SIZE(match_item) != 4)
        {
            PyErr_SetString(PyExc_ValueError, "invalid function arguments - expect tuples of size 4");
            Py_INCREF(Py_None);
            return Py_None;
        }

        int image_id_1 = (int)PyLong_AsLong(PyTuple_GET_ITEM(match_item, 0));
        int keypoint_id_1 = (int)PyLong_AsLong(PyTuple_GET_ITEM(match_item, 1));
        int image_id_2 = (int)PyLong_AsLong(PyTuple_GET_ITEM(match_item, 2));
        int keypoint_id_2 = (int)PyLong_AsLong(PyTuple_GET_ITEM(match_item, 3));

        match_s item = {
            {image_id_1, keypoint_id_1},
            {image_id_2, keypoint_id_2},
        };

        matches.push_back(item);
    }

    std::vector<cluster_s> clusters = build_clusters(matches);
    PyObject *clusters_list = PyList_New(clusters.size());
    int index = 0;
    for (cluster_s cluster : clusters)
    {
        PyObject *cluster_obj = PyList_New(cluster.items.size());
        int item_id = 0;
        for (item_s item : cluster.items)
        {
            PyObject *item_obj = PyTuple_New(2);
            PyTuple_SET_ITEM(item_obj, 0, PyLong_FromLong(item.frame_id));
            PyTuple_SET_ITEM(item_obj, 1, PyLong_FromLong(item.keypoint_id));
            PyList_SET_ITEM(cluster_obj, item_id, item_obj);
            item_id++;
        }
        PyList_SET_ITEM(clusters_list, index, cluster_obj);
        index++;
    }

    return clusters_list;
}

static PyMethodDef clusterization_methods[] = {
    {"build_clusters", (PyCFunction)build_clusters_from_matches, METH_VARARGS | METH_KEYWORDS,
     "Build clusters from match table"},
    {NULL} /* Sentinel */
};

static PyModuleDef clusterizationModule = {
    PyModuleDef_HEAD_INIT,
    .m_name = BASENAME,
    .m_doc = "Clusterization module",
    .m_size = -1,
    .m_methods = clusterization_methods,
};

PyMODINIT_FUNC
PyInit_clusterization(void)
{
    PyObject *m = PyModule_Create(&clusterizationModule);
    if (m == NULL)
        return NULL;

 
    return m;
}
