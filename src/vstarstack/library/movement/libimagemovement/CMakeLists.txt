cmake_minimum_required(VERSION 3.14)

project(imagemovement)

# Library sources
add_library(imagemovement STATIC src/flat.c
                                 src/sphere.c)

set(HEADERS "include/flat.h;include/sphere.h")
target_include_directories(imagemovement PUBLIC include)
set_target_properties(imagemovement PROPERTIES PUBLIC_HEADER "${HEADERS}")

# Fail on warnings
if(MSVC)
  target_compile_options(imagemovement PRIVATE /W4 /WX)
else()
  target_compile_options(imagemovement PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

install(TARGETS imagemovement
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libimagemovement)

# Tests
if (BUILD_TESTS)
    set(INSTALL_GTEST OFF)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    )

    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    add_executable(
        movement_test
        tests/movement_test.cc
    )

    target_link_libraries(
        movement_test
        GTest::gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(movement_test)
endif()
