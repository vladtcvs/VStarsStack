cmake_minimum_required(VERSION 3.10)

add_subdirectory(libimagedeform)

Python3_add_library(fine_movement MODULE
            module/imagedeform_gc.c
            module/imagedeform_lc.c
            module/imagedeform_module.c
            module/imagedeform.c
            module/imagegrid.c
            WITH_SOABI)

target_include_directories(fine_movement PUBLIC . lib "${NUMPY_INCLUDE_DIR}")
target_link_libraries(fine_movement PUBLIC imagedeform)

set_target_properties(fine_movement PROPERTIES OUTPUT_NAME "module")

install(TARGETS fine_movement
        COMPONENT python_modules
        DESTINATION "${PY_BUILD_CMAKE_IMPORT_NAME}/library/fine_movement")
