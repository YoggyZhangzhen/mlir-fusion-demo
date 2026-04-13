# lit 测试配置（可选，需要安装 lit 和 FileCheck）
find_program(FILECHECK_PATH FileCheck HINTS ${LLVM_TOOLS_BINARY_DIR})
find_program(LIT_PATH lit)

if(LIT_PATH AND FILECHECK_PATH)
  message(STATUS "Found lit: ${LIT_PATH}")
  message(STATUS "Found FileCheck: ${FILECHECK_PATH}")
  add_lit_testsuite(check-ryuki "Run RYUKI regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ryuki-opt ryuki-runner
  )
else()
  message(STATUS "lit or FileCheck not found, skipping test suite setup")
endif()
