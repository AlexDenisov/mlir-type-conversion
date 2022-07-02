#include "Lowering.h"
#include "FooDialect/FooDialect.h"

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Transforms/DialectConversion.h>

namespace lowering {
  struct ValueLowering : public mlir::ConversionPattern {
    ValueLowering(mlir::MLIRContext &context, mlir::TypeConverter &converter)
        : mlir::ConversionPattern(converter, foo::ValueOp::getOperationName(), 1, &context) {}

    mlir::LogicalResult matchAndRewrite(mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
      auto highOp = mlir::cast<foo::ValueOp>(op);
      auto retType = typeConverter->convertType(highOp.getType());
      auto lowOp = rewriter.create<mlir::ConstantOp>(highOp->getLoc(), mlir::IntegerAttr::get(retType, 42));
      rewriter.replaceOp(op, {lowOp});
      return mlir::success();
    }
  };

  struct BranchLowering : public mlir::ConversionPattern {
    BranchLowering(mlir::MLIRContext &context, mlir::TypeConverter &converter)
        : mlir::ConversionPattern(converter, foo::BranchOp::getOperationName(), 1, &context) {}

    mlir::LogicalResult matchAndRewrite(mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
      auto highOp = mlir::cast<foo::BranchOp>(op);
      auto lowOp = rewriter.create<mlir::BranchOp>(highOp.getLoc(), highOp.target(), operands);
      rewriter.eraseOp(highOp);
      return mlir::success();
    }
  };

  struct ReturnLowering : public mlir::ConversionPattern {
    ReturnLowering(mlir::MLIRContext &context, mlir::TypeConverter &converter)
        : mlir::ConversionPattern(converter, foo::ReturnOp::getOperationName(), 1, &context) {}

    mlir::LogicalResult matchAndRewrite(mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
      auto highOp = mlir::cast<foo::ReturnOp>(op);
      auto type = typeConverter->convertType(highOp.getType());
      auto lowOp = rewriter.create<mlir::ReturnOp>(highOp->getLoc(), type, operands);
      rewriter.replaceOp(highOp, { lowOp->getResult(0) });
      return mlir::success();
    }
  };

}// namespace lowering

Lowering::Lowering(mlir::MLIRContext &context) : context(context) {
}

bool Lowering::lower(mlir::ModuleOp module) {
  mlir::ConversionTarget target(context);
  target.addLegalOp<mlir::ModuleOp>();
  target.addLegalOp<mlir::FuncOp>();
  target.addLegalOp<mlir::ConstantOp>();
  target.addLegalOp<mlir::BranchOp>();
  target.addLegalOp<mlir::ReturnOp>();

  mlir::TypeConverter typeConverter;
  typeConverter.addConversion(
      [&](foo::foo_valueType type) -> llvm::Optional<mlir::Type> {
        return mlir::IntegerType::get(&context, 8);
      });

  mlir::RewritePatternSet patterns(&context);
  patterns.add<
      //
      lowering::ValueLowering,
      lowering::BranchLowering,
      lowering::ReturnLowering

      //
      >(context, typeConverter);

  mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  if (mlir::failed(mlir::applyPartialConversion(module.getOperation(), target, frozenPatterns))) {
    //  if (mlir::failed(mlir::applyFullConversion(module.getOperation(), target, frozenPatterns))) {
    llvm::errs() << "Cannot apply conversion\n";
    return false;
  }
  return true;
}
