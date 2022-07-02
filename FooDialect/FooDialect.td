include "mlir/IR/OpBase.td"

def Foo_Dialect : Dialect {
  let name = "foo";
  let summary = "Foo Dialect";

  let cppNamespace = "foo";
}

class FooType<string name> : TypeDef<Foo_Dialect, name> {
  let summary = name;
}

def FooValueType : FooType<"foo_value"> {}

class Foo_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Foo_Dialect, mnemonic, traits>;

def ValueOp : Foo_Op<"value"> {
  let summary = "value";
  let results = (outs FooValueType);
}

def BranchOp : Foo_Op<"br", [Terminator]> {
  let summary = "br";
  let arguments = (ins FooValueType);
  let successors = (successor AnySuccessor:$target);
}

def ReturnOp : Foo_Op<"return", [Terminator]> {
  let summary = "return";
  let arguments = (ins FooValueType);
  let results = (outs FooValueType);
}
