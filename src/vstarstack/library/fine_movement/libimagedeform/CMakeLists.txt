cmake_minimum_required(VERSION 3.14)

project(imagedeform)

# Library sources
add_library(imagedeform STATIC src/interpolation.c
                               src/image_grid.c
                               src/image_deform.c
                               src/image_deform_gc.c
                               src/image_deform_lc.c)

set(HEADERS "include/interpolation.h;"
            "include/image_grid.h;"
            "include/image_deform.h;"
            "include/image_deform_gc.h;"
            "include/image_deform_lc.h")

target_include_directories(imagedeform PUBLIC include)
set_target_properties(imagedeform PROPERTIES PUBLIC_HEADER "${HEADERS}")

# Fail on warnings
if(MSVC)
  target_compile_options(imagedeform PRIVATE /W4 /WX)
else()
  target_compile_options(imagedeform PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

install(TARGETS imagedeform
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libimagedeform)

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
        deform_test
        tests/deform_test.cc
    )

    target_link_libraries(
        deform_test
        GTest::gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(deform_test)
endif()
