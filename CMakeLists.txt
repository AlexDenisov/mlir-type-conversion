cmake_minimum_required(VERSION 3.22)
project(FooBar)

set(CMAKE_CXX_STANDARD 17)

find_package(LLVM CONFIG REQUIRED)
message("Using LLVM_CONFIG: ${LLVM_CONFIG}")
find_package(MLIR CONFIG REQUIRED)
message("Using MLIR_CONFIG: ${MLIR_CONFIG}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_DIR}")
include(AddLLVM)
include(TableGen)
include(AddMLIR)

add_subdirectory(FooDialect)
add_executable(FooBar main.cpp Lowering.cpp Lowering.h)
target_link_libraries(FooBar FooDialect MLIRStandard MLIRTransforms)
