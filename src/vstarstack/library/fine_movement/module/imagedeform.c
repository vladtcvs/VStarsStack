#include "imagedeform.h"

/**
 * \brief Init ImageDeform
 * \param _self ImageDeformObject object
 * \param args arguments
 * \param kwads named arguments
 * \return 0 for OK
 */
static int ImageDeform_init(PyObject *_self, PyObject *args, PyObject *kwds)
{
    int image_w, image_h;
    int grid_w, grid_h;
    struct ImageDeformObject *self = (struct ImageDeformObject *)_self;
    static char *kwlist[] = {"image_w", "image_h", "grid_w", "grid_h", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiii", kwlist,
                                     &image_w, &image_h, &grid_w, &grid_h))
        return -1;

    return image_deform_init(&self->deform, grid_w, grid_h, image_w, image_h);
}

static void ImageDeform_finalize(PyObject *_self)
{
    PyObject *error_type, *error_value, *error_traceback;

    /* Save the current exception, if any. */
    PyErr_Fetch(&error_type, &error_value, &error_traceback);

    struct ImageDeformObject *self = (struct ImageDeformObject *)_self;
    image_deform_finalize(&self->deform);

    /* Restore the saved exception. */
    PyErr_Restore(error_type, error_value, error_traceback);
}
