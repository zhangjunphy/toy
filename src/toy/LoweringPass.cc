#include "toy/Dialect.h"

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

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
      : ConversionPattern(BinaryOp::getOperationName(), 1, context) {}

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

struct FuncOpLowering : public OpConversionPattern<FuncOp> {
  using OpConversionPattern<FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    if (op.getName() != "main")
      return failure();
    if (op.getNumArguments() || op.getFunctionType().getNumResults()) {
      return rewriter.notifyMatchFailure(op, [](Diagnostic &diag) {
        diag << "unexpected 'main' function type";
      });
    }
    auto func =
        rewriter.create<func::FuncOp>(loc, op.getName(), op.getFunctionType());

    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return success();
  }
};

struct PrintOpLowering : public OpConversionPattern<PrintOp> {
  using OpConversionPattern<PrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.updateRootInPlace(op,
                               [&] { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

struct ReturnOpLowering : public OpRewritePattern<ReturnOp> {
  using OpRewritePattern<ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReturnOp op,
                                PatternRewriter &rewriter) const final {
    if (op.hasOperand())
      return failure();
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return success();
  }
};

struct ToyToAffineLoweringPass
    : public PassWrapper<ToyToAffineLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToAffineLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, func::FuncDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override final {
    mlir::ConversionTarget target(getContext());

    target.addLegalDialect<AffineDialect, arith::ArithDialect,
                           func::FuncDialect, memref::MemRefDialect>();
    target.addIllegalDialect<ToyDialect>();
    target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
      return llvm::none_of(op->getOperandTypes(),
                           [](Type type) { return type.isa<TensorType>(); });
    });

    RewritePatternSet patterns(&getContext());
    patterns.add<BinaryOpLowering<AddOp, arith::AddFOp>,
                 BinaryOpLowering<MulOp, arith::MulFOp>, ConstantOpLowering,
                 TransposeOpLowering, FuncOpLowering, PrintOpLowering,
                 ReturnOpLowering>(&getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

struct PrintOpToLLVMLowering : public OpConversionPattern<PrintOp> {
  using OpConversionPattern<PrintOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto memRefType = (*op->operand_type_begin()).cast<MemRefType>();
    auto memRefShape = memRefType.getShape();
    auto loc = op.getLoc();

    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto printf = getOrInsertPrintf(rewriter, module);
    Value formatSpecifierCst = getOrCreateGlobalString(
        loc, rewriter, "frmt_spec", StringRef("%f \0", 4), module);
    Value newLineCst = getOrCreateGlobalString(loc, rewriter, "nl",
                                               StringRef("\n\0", 2), module);

    SmallVector<Value, 4> loopIvs;
    for (unsigned i = 0, e = memRefShape.size(); i != e; ++i) {
      auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto upperBound =
          rewriter.create<arith::ConstantIndexOp>(loc, memRefShape[i]);
      auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      auto loop =
          rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
      for (Operation &nested : *loop.getBody()) {
        rewriter.eraseOp(&nested);
      }
      loopIvs.push_back(loop.getInductionVar());
      rewriter.setInsertionPointToEnd(loop.getBody());
      if (i != e - 1)
        rewriter.create<func::CallOp>(loc, printf, rewriter.getIntegerType(32),
                                      newLineCst);
      rewriter.create<scf::YieldOp>(loc);
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    auto printOp = cast<toy::PrintOp>(op);
    auto elementLoad =
        rewriter.create<memref::LoadOp>(loc, printOp.getInput(), loopIvs);
    rewriter.create<func::CallOp>(
        loc, printf, rewriter.getIntegerType(32),
        ArrayRef<Value>{{formatSpecifierCst, elementLoad}});

    rewriter.eraseOp(op);
    return success();
  }

private:
  static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                             ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf")) {
      return SymbolRefAttr::get(context, "printf");
    }

    auto llvmI32Ty = IntegerType::get(context, 32);
    auto llvmI8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
    auto llvmFnTy = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy, true);

    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnTy);
    return SymbolRefAttr::get(context, "printf");
  }

  static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                       StringRef name, StringRef value,
                                       ModuleOp module) {
    LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = LLVM::LLVMArrayType::get(
          IntegerType::get(builder.getContext(), 8), value.size());
      global = builder.create<LLVM::GlobalOp>(loc, type, true,
                                              LLVM::Linkage::Internal, name,
                                              builder.getStringAttr(value), 0);
    }

    Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
    Value cst0 = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                  builder.getIndexAttr(0));

    return builder.create<LLVM::GEPOp>(
        loc,
        LLVM::LLVMPointerType::get(IntegerType::get(builder.getContext(), 8)),
        globalPtr, ArrayRef<Value>{{cst0, cst0}});
  }
};

struct AffineToLLVMLoweringPass
    : public PassWrapper<AffineToLLVMLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AffineToLLVMLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
  }

  void runOnOperation() override final {
    mlir::ConversionTarget target(getContext());

    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalOp<mlir::ModuleOp>();

    mlir::LLVMTypeConverter typeConverter(&getContext());
    mlir::RewritePatternSet patterns(&getContext());
    mlir::populateAffineToStdConversionPatterns(patterns);
    arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateSCFToControlFlowConversionPatterns(patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
    patterns.add<PrintOpToLLVMLowering>(&getContext());

    mlir::ModuleOp module = getOperation();
    if (mlir::failed(
            mlir::applyFullConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createLoweringPass() {
  return std::make_unique<ToyToAffineLoweringPass>();
}

std::unique_ptr<mlir::Pass> createLoweringToLLVMPass() {
  return std::make_unique<AffineToLLVMLoweringPass>();
}
} // namespace toy
} // namespace mlir
