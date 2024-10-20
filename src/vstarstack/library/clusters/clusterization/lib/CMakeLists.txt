cmake_minimum_required(VERSION 3.14)

project(clusters)

# Library sources
add_library(clusters STATIC src/clusters.cc)

set(HEADERS "include/clusters.hpp;")

target_include_directories(clusters PUBLIC include)
set_target_properties(clusters PROPERTIES PUBLIC_HEADER "${HEADERS}")

# Fail on warnings
if(MSVC)
  target_compile_options(clusters PRIVATE /W4 /WX)
else()
  target_compile_options(clusters PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

install(TARGETS clusters
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libclusters)

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
        clusters_test
        tests/clusters_test.cc
    )

    target_link_libraries(
        clusters_test
        GTest::gtest_main
        clusters
    )

    include(GoogleTest)
    gtest_discover_tests(clusters_test)

    if (UNIX)
        add_custom_target(memory_leak_test sh ${CMAKE_SOURCE_DIR}/run_valgrind_tests.sh ${CMAKE_BINARY_DIR}/clusters_test
                          DEPENDS clusters_test)
    endif()
endif()
