#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "toy/Dialect.h"

// namespace {
// #include "Optimize.inc"
// }

struct SimplifyRedundantTranspose
    : public mlir::OpRewritePattern<mlir::toy::TransposeOp> {
  SimplifyRedundantTranspose(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<mlir::toy::TransposeOp>(context, 1) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::toy::TransposeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value transposeInput = op.getOperand();
    mlir::toy::TransposeOp transposeInputOp =
        transposeInput.getDefiningOp<mlir::toy::TransposeOp>();

    if (!transposeInputOp)
      return mlir::failure();

    rewriter.replaceOp(op, {transposeInputOp.getOperand()});
    return mlir::success();
  }
};

void mlir::toy::TransposeOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<SimplifyRedundantTranspose>(context);
}
