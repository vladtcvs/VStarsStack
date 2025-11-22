cmake_minimum_required(VERSION 3.10)

Python3_add_library(movements MODULE module.c lib/flat.c lib/sphere.c WITH_SOABI)

target_include_directories(movements PUBLIC . lib "${NUMPY_INCLUDE_DIR}")
target_link_libraries(movements PUBLIC proj projections_structs)

install(TARGETS movements
        COMPONENT python_modules
        DESTINATION ${PY_BUILD_CMAKE_IMPORT_NAME}/library/movement)
