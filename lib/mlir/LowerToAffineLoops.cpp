#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "toy/Dialect.h"
#include "toy/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include <stdint.h>

namespace {
struct ToyToAffineLoweringPass
    : public mlir::PassWrapper<ToyToAffineLoweringPass, mlir::FunctionPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::AffineDialect, mlir::memref::MemRefDialect,
                    mlir::StandardOpsDialect>();
  }

  void runOnFunction() final;
};
} // namespace

static mlir::MemRefType convertTensorToMemRef(mlir::TensorType type) {
  assert(type.hasRank() && "expected only ranked shapes");
  return mlir::MemRefType::get(type.getShape(), type.getElementType());
}

static mlir::Value insertAllocAndDealloc(mlir::MemRefType type,
                                         mlir::Location loc,
                                         mlir::PatternRewriter &rewriter) {
  auto alloc = rewriter.create<mlir::memref::AllocOp>(loc, type);
  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  auto dealloc = rewriter.create<mlir::memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}

using LoopIterationFn = llvm::function_ref<mlir::Value(
    mlir::OpBuilder &rewriter, mlir::ValueRange memRefOperands,
    mlir::ValueRange loopLvs)>;

static void lowerOpToLoops(mlir::Operation *op, mlir::ValueRange operands,
                           mlir::PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {
  auto tensorType = (*op->result_type_begin()).cast<mlir::TensorType>();
  auto loc = op->getLoc();

  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

  mlir::SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), 0);
  mlir::SmallVector<int64_t, 4> steps(tensorType.getRank(), 1);
  mlir::buildAffineLoopNest(
      rewriter, loc, lowerBounds, tensorType.getShape(), steps,
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc,
          mlir::ValueRange ivs) {
        mlir::Value valueToStore =
            processIteration(nestedBuilder, operands, ivs);
        nestedBuilder.create<mlir::AffineStoreOp>(loc, valueToStore, alloc,
                                                  ivs);
      });

  rewriter.replaceOp(op, alloc);
}

namespace {

template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public mlir::ConversionPattern {
  BinaryOpLowering(mlir::MLIRContext *ctx)
      : mlir::ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(
        op, operands, rewriter,
        [loc](mlir::OpBuilder &builder, mlir::ValueRange memRefOperands,
              mlir::ValueRange loopIvs) {
          typename BinaryOp::Adaptor adaptor(memRefOperands);

          auto loopLhs =
              builder.create<mlir::AffineLoadOp>(loc, adaptor.lhs(), loopIvs);
          auto loopRhs =
              builder.create<mlir::AffineLoadOp>(loc, adaptor.rhs(), loopIvs);

          return builder.create<LoweredBinaryOp>(loc, loopLhs, loopRhs);
        });

    return mlir::success();
  }
};

using AddOpLowering = BinaryOpLowering<mlir::toy::AddOp, mlir::arith::AddFOp>;
using MulOpLowering = BinaryOpLowering<mlir::toy::MulOp, mlir::arith::MulFOp>;

struct ConstantOpLowering
    : public mlir::OpRewritePattern<mlir::toy::ConstantOp> {
  using mlir::OpRewritePattern<mlir::toy::ConstantOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::toy::ConstantOp op,
                  mlir::PatternRewriter &rewriter) const final {
    mlir::DenseElementsAttr constantValue = op.value();
    mlir::Location loc = op.getLoc();

    auto tensorType = op.getType().cast<mlir::TensorType>();
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    auto valueShape = memRefType.getShape();
    mlir::SmallVector<mlir::Value, 8> constantIndices;

    if (!valueShape.empty()) {
      for (auto i : llvm::seq<int64_t>(
               0, *std::max_element(valueShape.begin(), valueShape.end())))
        constantIndices.push_back(
            rewriter.create<mlir::arith::ConstantIndexOp>(loc, i));
    } else {
      constantIndices.push_back(
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0));
    }

    mlir::SmallVector<mlir::Value, 2> indices;
    auto valueIt = constantValue.value_begin<mlir::FloatAttr>();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      if (dimension == valueShape.size()) {
        rewriter.create<mlir::AffineStoreOp>(
            loc, rewriter.create<mlir::arith::ConstantOp>(loc, *valueIt++),
            alloc, llvm::makeArrayRef(indices));
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
    return mlir::success();
  }
};

struct PrintOpLowering : public mlir::OpConversionPattern<mlir::toy::PrintOp> {
  using mlir::OpConversionPattern<mlir::toy::PrintOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::toy::PrintOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    rewriter.updateRootInPlace(op,
                               [&] { op->setOperands(adaptor.getOperands()); });
    return mlir::success();
  }
};

struct ReturnOpLowering : public mlir::OpRewritePattern<mlir::toy::ReturnOp> {
  using mlir::OpRewritePattern<mlir::toy::ReturnOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::toy::ReturnOp op,
                  mlir::PatternRewriter &rewriter) const final {
    if (op.hasOperand())
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::ReturnOp>(op);
    return mlir::success();
  }
};

struct TransposeOpLowering : public mlir::ConversionPattern {
  TransposeOpLowering(mlir::MLIRContext *ctx)
      : mlir::ConversionPattern(mlir::toy::TransposeOp::getOperationName(), 1,
                                ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    lowerOpToLoops(
        op, operands, rewriter,
        [loc](mlir::OpBuilder &builder, mlir::ValueRange memRefOperands,
              mlir::ValueRange loopIvs) {
          mlir::toy::TransposeOpAdaptor adaptor(memRefOperands);
          mlir::Value input = adaptor.input();

          mlir::SmallVector<mlir::Value, 2> reserveIvs(llvm::reverse(loopIvs));
          return builder.create<mlir::AffineLoadOp>(loc, input, reserveIvs);
        });
    return mlir::success();
  }
};
} // namespace

void ToyToAffineLoweringPass::runOnFunction() {
  using namespace mlir;

  ConversionTarget target(getContext());
  target.addLegalDialect<AffineDialect, arith::ArithmeticDialect,
                         memref::MemRefDialect, StandardOpsDialect>();
  target.addIllegalDialect<toy::ToyDialect>();
  target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
    return llvm::none_of(op->getOperandTypes(),
                         [](Type type) { return type.isa<TensorType>(); });
  });

  RewritePatternSet patterns(&getContext());
  patterns.add<AddOpLowering, MulOpLowering, ConstantOpLowering,
               ReturnOpLowering, PrintOpLowering, TransposeOpLowering>(
      &getContext());
  if (failed(
          applyPartialConversion(getFunction(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::toy::createLowerToAffinePass() {
  return std::make_unique<ToyToAffineLoweringPass>();
}
