cmake_minimum_required(VERSION 3.4)
project(Device)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/home/henry/anaconda2/envs/env_ardennes/include/python3.6m")

set(SOURCE_FILES cpu_code.cpp)
set(PYTHON_FILES __test_implementation__.py)

add_executable(Device ${SOURCE_FILES} ${PYTHON_FILES})
