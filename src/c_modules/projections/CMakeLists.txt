cmake_minimum_required(VERSION 3.10)

add_library(proj STATIC lib/equirectangular.c lib/orthographic.c lib/perspective.c)

add_library(projections_structs INTERFACE)
target_include_directories(projections_structs INTERFACE .)

Python3_add_library(projections MODULE module.c WITH_SOABI)

target_include_directories(projections PUBLIC lib "${NUMPY_INCLUDE_DIR}")
target_link_libraries(projections PUBLIC proj projections_structs)

install(TARGETS projections
        COMPONENT python_modules
        DESTINATION ${PY_BUILD_CMAKE_IMPORT_NAME}/library/projection)
