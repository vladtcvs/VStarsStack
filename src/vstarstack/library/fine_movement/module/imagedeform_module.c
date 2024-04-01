/*
 * Copyright (c) 2022-2024 Vladislav Tsendrovskii
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

#include "imagegrid.h"
#include "imagedeform.h"
#include "imagedeform_gc.h"
#include "imagedeform_lc.h"

static PyModuleDef image_deformModule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "vstarstack.library.fine_movement.module",
    .m_doc = "Image deform module for fine images matching",
    .m_size = -1,
};

PyMODINIT_FUNC
PyInit_module(void)
{
    PyObject *m;
    if (PyType_Ready(&ImageGrid) < 0)
        return NULL;
    if (PyType_Ready(&ImageDeform) < 0)
        return NULL;
    if (PyType_Ready(&ImageDeformGC) < 0)
        return NULL;
    if (PyType_Ready(&ImageDeformLC) < 0)
        return NULL;

    m = PyModule_Create(&image_deformModule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&ImageGrid);
    if (PyModule_AddObject(m, "ImageGrid", (PyObject *)&ImageGrid) < 0)
    {
        Py_DECREF(&ImageGrid);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&ImageDeform);
    if (PyModule_AddObject(m, "ImageDeform", (PyObject *)&ImageDeform) < 0)
    {
        Py_DECREF(&ImageGrid);
        Py_DECREF(&ImageDeform);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&ImageDeformGC);
    if (PyModule_AddObject(m, "ImageDeformGC", (PyObject *)&ImageDeformGC) < 0)
    {
        Py_DECREF(&ImageGrid);
        Py_DECREF(&ImageDeform);
        Py_DECREF(&ImageDeformGC);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&ImageDeformLC);
    if (PyModule_AddObject(m, "ImageDeformLC", (PyObject *)&ImageDeformLC) < 0)
    {
        Py_DECREF(&ImageGrid);
        Py_DECREF(&ImageDeform);
        Py_DECREF(&ImageDeformGC);
        Py_DECREF(&ImageDeformLC);
        Py_DECREF(m);
        return NULL;
    }

    import_array();
    return m;
}
