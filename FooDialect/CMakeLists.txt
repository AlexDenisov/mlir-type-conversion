add_library(FooDialect
  FooDialect.h
  FooDialect.cpp
)
target_link_libraries(FooDialect LLVMSupport MLIRDialect)
target_include_directories(FooDialect PUBLIC ${MLIR_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
target_compile_options(FooDialect PUBLIC -fno-rtti -fvisibility=hidden)
include_directories(${MLIR_INCLUDE_DIRS})

set(LLVM_TARGET_DEFINITIONS FooDialect.td)
mlir_tablegen(FooOps.h.inc -gen-op-decls)
mlir_tablegen(FooOps.cpp.inc -gen-op-defs)
mlir_tablegen(FooDialect.h.inc -gen-dialect-decls)
mlir_tablegen(FooDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(FooTypeDefs.h.inc -gen-typedef-decls)
mlir_tablegen(FooTypeDefs.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(FooDialectGen)
add_dependencies(FooDialect FooDialectGen)
