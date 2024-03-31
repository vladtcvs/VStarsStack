cmake_minimum_required(VERSION 3.14)

project(imagedeform)

add_library(imagedeform STATIC src/interpolation.c
                               src/image_grid.c
                               src/image_deform.c
                               src/image_deform_gc.c
                               src/image_deform_lc.c)

target_include_directories(imagedeform PUBLIC include)

if(MSVC)
  target_compile_options(imagedeform PRIVATE /W4 /WX)
else()
  target_compile_options(imagedeform PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()
