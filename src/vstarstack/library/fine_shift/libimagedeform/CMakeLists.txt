project(libimagedeform)

add_library(imagedeform STATIC
            src/interpolation.c
            src/image_grid.c
            src/image_deform.c
            src/image_deform_gc.c)

target_include_directories(imagedeform PUBLIC include)
