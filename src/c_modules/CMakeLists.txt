cmake_minimum_required(VERSION 3.10)
project(vstarstack)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

message("Python: '${Python3_EXECUTABLE}'")
execute_process(COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/get_numpy_path.py"
                OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
                RESULT_VARIABLE NUMPY_NOT_FOUND
                )
if(NUMPY_NOT_FOUND)
    message(FATAL_ERROR "NumPy headers not found: ${NUMPY_NOT_FOUND}")
endif()

message("Numpy headers: '${NUMPY_INCLUDE_DIR}'")

add_subdirectory(projections)
add_subdirectory(movements)
add_subdirectory(fine_movement)
add_subdirectory(clusterization)
add_subdirectory(bayes)
