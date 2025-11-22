cmake_minimum_required(VERSION 3.10)

add_library(bayes STATIC bayes.c)
target_include_directories(bayes INTERFACE .)
