#include "toy/Dialect.h"

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir {
namespace toy {

static MemRefType convertTensorToMemRef(TensorType type) {
  assert(type.hasRank() && "expected only ranked shapes");
  return MemRefType::get(type.getShape(), type.getElementType());
}

static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
  auto alloc = rewriter.create<memref::AllocOp>(loc, type);
  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}

using LoopIterationFn = function_ref<Value(
    OpBuilder &rewriter, ValueRange memRefOperands, ValueRange loopIvs)>;
static void lowerOpToLoops(Operation *op, ValueRange operands,
                           PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {
  auto tensorType = (*op->result_type_begin()).cast<TensorType>();
  auto loc = op->getLoc();

  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);
  SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
  SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);
  buildAffineLoopNest(
      rewriter, loc, lowerBounds, tensorType.getShape(), steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        Value valueToStore = processIteration(nestedBuilder, operands, ivs);
        nestedBuilder.create<AffineStoreOp>(loc, valueToStore, alloc, ivs);
      });

  rewriter.replaceOp(op, alloc);
}

template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public ConversionPattern {
  BinaryOpLowering(MLIRContext *context)
      : ConversionPattern(BinaryOp::getOperationname(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(
        op, operands, rewriter,
        [loc](OpBuilder &rewriter, ValueRange memRefOperands,
              ValueRange loopIvs) {
          typename BinaryOp::Adaptor adaptor(memRefOperands);

          auto loadLhs =
              rewriter.create<AffineLoadOp>(loc, adaptor.getLhs(), loopIvs);
          auto loadRhs =
              rewriter.create<AffineLoadOp>(loc, adaptor.getRhs(), loopIvs);

          return rewriter.create<LoweredBinaryOp>(loc, loadLhs, loadRhs);
        });

    return success();
  }
};

struct TransposeOpLowering : public ConversionPattern {
  TransposeOpLowering(MLIRContext *context)
      : mlir::ConversionPattern(TransposeOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    lowerOpToLoops(op, operands, rewriter,
                   [loc](OpBuilder &rewriter, ValueRange memRefOperands,
                         ValueRange loopIvs) {
                     TransposeOpAdaptor transposeAdaptor(memRefOperands);
                     auto input = transposeAdaptor.getInput();
                     SmallVector<Value, 2> reverseIvs(llvm::reverse(loopIvs));
                     return rewriter.create<AffineLoadOp>(loc, input,
                                                          reverseIvs);
                   });

    return success();
  }
};

struct ConstantOpLowering : public OpRewritePattern<ConstantOp> {
  using OpRewritePattern<ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConstantOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    DenseElementsAttr value = op.getValue();
    auto tensorType = op.getType().cast<TensorType>();
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    auto valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;
    if (!valueShape.empty()) {
      for (auto i : llvm::seq<int64_t>(
               0, *std::max_element(valueShape.begin(), valueShape.end()))) {
        constantIndices.push_back(
            rewriter.create<arith::ConstantIndexOp>(loc, i));
      }
    } else {
      constantIndices.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, 0));
    }

    SmallVector<Value, 2> indices;
    auto valueIt = value.value_begin<FloatAttr>();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      if (dimension == valueShape.size()) {
        rewriter.create<AffineStoreOp>(
            loc, rewriter.create<arith::ConstantOp>(loc, *valueIt++), alloc,
            llvm::ArrayRef(indices));
        return;
      }

      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    storeElements(0);

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

class ToyToAffineLoweringPass
    : public PassWrapper<ToyToAffineLoweringPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());

    target.addLegalDialect<AffineDialect, arith::ArithDialect,
                           func::FuncDialect, memref::MemRefDialect>();
    target.addIllegalDialect<ToyDialect>();
    target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
      return llvm::none_of(op->getOperandTypes(),
                           [](Type type) { return type.isa<TensorType>(); });
    });

    RewritePatternSet patterns(&getContext());
    patterns.add<TransposeOpLowering>(&getContext());
  }
};

} // namespace toy
} // namespace mlir
