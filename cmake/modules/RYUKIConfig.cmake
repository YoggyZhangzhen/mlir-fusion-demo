# RYUKI 项目的 CMake 辅助模块
# 可在此添加项目级别的宏和函数

# 添加一个 RYUKI 库的便捷宏（封装 add_mlir_library 的常用配置）
macro(add_ryuki_library name)
  add_mlir_library(${name} ${ARGN})
endmacro()
