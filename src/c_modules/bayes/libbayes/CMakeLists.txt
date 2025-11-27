cmake_minimum_required(VERSION 3.10)
project(bayes)

add_library(bayes STATIC bayes.c)
target_include_directories(bayes INTERFACE .)

if (BUILD_TESTS)
    enable_testing()
    include(CTest)
    add_subdirectory(tests)
endif()
