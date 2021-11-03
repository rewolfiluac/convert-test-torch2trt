cmake_minimum_required(VERSION 3.14)

project(preproc)

cmake_policy(SET CMP0078 NEW)
cmake_policy(SET CMP0086 NEW)

find_package(PythonLibs 3.8 REQUIRED)
find_package(Python3 3.8 COMPONENTS Interpreter NumPy REQUIRED)
find_package(SWIG REQUIRED)
find_package(OpenMP REQUIRED)

include(${SWIG_USE_FILE})
include(GenerateExportHeader)

set(CMAKE_C_FLAGS "\${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

set(SRCS
    preproc.cpp
    preproc.h
)

set(INTERFACE_FILES
    preproc.i
)

set_source_files_properties(${INTERFACE_FILES} PROPERTIES CPLUSPLUS ON)

swig_add_library(${PROJECT_NAME}
    LANGUAGE python
    SOURCES
    ${INTERFACE_FILES} ${SRCS}
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PYTHON_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${PYTHON_LIBRARIES} Python3::NumPy
)
