#pragma once

#include <mlir/IR/BuiltinOps.h>

class Lowering {
public:
  explicit Lowering(mlir::MLIRContext &context);
  bool lower(mlir::ModuleOp module);

private:
  mlir::MLIRContext &context;
};
