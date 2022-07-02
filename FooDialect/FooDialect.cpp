#include "FooDialect.h"

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

using namespace mlir;
using namespace foo;

#include "FooDialect.cpp.inc"

void FooDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "FooOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "FooTypeDefs.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "FooOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "FooTypeDefs.cpp.inc"

mlir::Type FooDialect::parseType(mlir::DialectAsmParser &parser) const {
  auto context = parser.getBuilder().getContext();
  if (parser.parseKeyword("foo_value", "Expect foo type")) {
    return foo_valueType::get(context);
  }
  return nullptr;
}

void FooDialect::printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const {
  if (type.isa<foo_valueType>()) {
    printer << "foo_value";
  }
}
