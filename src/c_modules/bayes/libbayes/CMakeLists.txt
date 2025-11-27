cmake_minimum_required(VERSION 3.10)
project(bayes)

if (BUILD_TESTS)
    enable_testing()
    include(CTest)
endif()

add_library(bayes STATIC bayes.c)
target_include_directories(bayes INTERFACE .)

add_subdirectory(tests)
