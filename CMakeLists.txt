cmake_minimum_required(VERSION 3.20)
project(RYUKI LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ─── 找到 LLVM ────────────────────────────────────────────────────────────────
# 使用方式: cmake .. -DLLVM_DIR=/path/to/llvm-install/lib/cmake/llvm
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "LLVM install prefix: ${LLVM_INSTALL_PREFIX}")

# ─── 找到 MLIR ────────────────────────────────────────────────────────────────
# 使用方式: cmake .. -DMLIR_DIR=/path/to/llvm-install/lib/cmake/mlir
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found MLIR in: ${MLIR_DIR}")

# ─── 加载 LLVM/MLIR CMake 工具模块 ───────────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# ─── 头文件 & 宏定义 ──────────────────────────────────────────────────────────
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)   # TableGen 生成的 .inc 文件

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

# ─── 子目录 ───────────────────────────────────────────────────────────────────
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)
