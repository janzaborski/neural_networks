cmake_minimum_required(VERSION 3.30)
project(hopfield LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Boost 1.92 CONFIG REQUIRED COMPONENTS json program_options unit_test_framework)
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)

add_library(hopfield_options INTERFACE)
target_compile_features(hopfield_options INTERFACE cxx_std_26)
target_compile_options(hopfield_options INTERFACE
    -freflection
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow
)

add_subdirectory(src)
add_subdirectory(app)

enable_testing()
add_subdirectory(tests)
