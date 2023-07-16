#pragma once

#include "toy/Dialect.h"

#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::toy;

struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
  SimplifyRedundantTranspose(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<TransposeOp>(context) {}

  mlir::LogicalResult
  matchAndRewrite(TransposeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value input = op.getOperand();
    TransposeOp inputOp = input.getDefiningOp<TransposeOp>();

    if (!inputOp)
      return failure();

    rewriter.replaceOp(op, {inputOp.getOperand()});
    return success();
  }
};
