cmake_minimum_required(VERSION 3.10)

add_subdirectory(lib)

Python3_add_library(clusterization MODULE
            module.cc
            WITH_SOABI)

target_include_directories(clusterization PUBLIC . lib)
target_link_libraries(clusterization PUBLIC clusters)

install(TARGETS clusterization
        COMPONENT python_modules
        DESTINATION "${PY_BUILD_CMAKE_IMPORT_NAME}/library/clusters")
