cmake_minimum_required(VERSION 3.10)

add_subdirectory(libbayes)

Python3_add_library(bayes_py MODULE module.c WITH_SOABI)

target_include_directories(bayes_py PUBLIC lib "${NUMPY_INCLUDE_DIR}")
target_link_libraries(bayes_py PUBLIC bayes)
set_target_properties(bayes_py PROPERTIES OUTPUT_NAME "bayes")

install(TARGETS bayes_py
        COMPONENT python_modules
        DESTINATION ${PY_BUILD_CMAKE_IMPORT_NAME}/library/bayes)
