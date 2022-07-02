#pragma once

#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "FooDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "FooTypeDefs.h.inc"

#define GET_OP_CLASSES
#include "FooOps.h.inc"
