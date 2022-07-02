#include "FooDialect/FooDialect.h"
#include "Lowering.h"

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>

int main() {
  mlir::DialectRegistry registry;
  registry.insert<foo::FooDialect>();
  registry.insert<mlir::StandardOpsDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto location = mlir::UnknownLoc::get(&context);

  mlir::OpBuilder builder(&context);
  mlir::ModuleOp module = mlir::ModuleOp::create(location, llvm::StringRef("a_module"));

  auto foo_value_t = foo::foo_valueType::get(&context);
  auto functionType = builder.getFunctionType({}, { foo_value_t });
  auto function = builder.create<mlir::FuncOp>(location, "f", functionType);

  auto entry = function.addEntryBlock();
  auto exit = function.addBlock();
  exit->addArgument(foo_value_t);

  builder.setInsertionPointToStart(entry);
  auto value = builder.create<foo::ValueOp>(location, foo_value_t);
  auto branch = builder.create<foo::BranchOp>(location, value, exit);

  builder.setInsertionPointToStart(exit);
  builder.create<foo::ReturnOp>(location, foo_value_t, exit->getArgument(0));

  module.push_back(function);

  module.print(llvm::errs());
  llvm::errs() << "\n";

  if (mlir::verify(module).failed()) {
    llvm::errs() << "Invalid module\n";
    return 1;
  }

  Lowering lowering(context);
  if (!lowering.lower(module)) {
    llvm::errs() << "Lowering failed\n";
    return 1;
  }

  if (mlir::verify(module).failed()) {
    llvm::errs() << "Invalid module after lowering\n";
//    return 1;
  }

  module.print(llvm::errs());
  llvm::errs() << "\n";

  return 0;
}
