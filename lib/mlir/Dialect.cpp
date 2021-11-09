#include "toy/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "toy/Dialect.cpp.inc"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/SMLoc.h"

void mlir::toy::ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "toy/Ops.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ConstantOp
void mlir::toy::ConstantOp::build(mlir::OpBuilder &builder,
                                  mlir::OperationState &state, double value) {
  auto dataType = mlir::RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = mlir::DenseElementsAttr::get(dataType, value);
  ConstantOp::build(builder, state, dataType, dataAttribute);
}

static mlir::ParseResult parseConstantOp(mlir::OpAsmParser &parser,
                                         mlir::OperationState &result) {
  mlir::DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes))
    return mlir::failure();

  result.addTypes(value.getType());
  return mlir::success();
}

static void print(mlir::OpAsmPrinter &printer, mlir::toy::ConstantOp op) {
  printer << " ";
  printer.printOptionalAttrDict(op->getAttrs(), {"value"});
  printer << op.value();
}

static mlir::LogicalResult verify(mlir::toy::ConstantOp op) {
  auto resultType = op.getResult().getType().dyn_cast<mlir::RankedTensorType>();
  if (!resultType)
    return mlir::success();

  auto attrType = op.value().getType().cast<mlir::TensorType>();
  if (attrType.getRank() != resultType.getRank()) {
    return op.emitOpError("return type must match the one of the attached "
                          "value attribute: ")
           << attrType.getRank() << " != " << resultType.getRank();
  }

  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
    if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
      return op.emitOpError(
                 "result type shape mismatches its attribute at dimension")
             << dim << ": " << attrType.getShape()[dim]
             << " != " << resultType.getShape()[dim];
    }
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// BinaryOps
static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  mlir::SmallVector<mlir::OpAsmParser::OperandType, 2> operands;
  llvm::SMLoc operandsloc = parser.getCurrentLocation();
  mlir::Type type;

  if (parser.parseOperandList(operands, 2) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type))
    return mlir::failure();

  if (mlir::FunctionType funcType = type.dyn_cast<mlir::FunctionType>()) {
    if (parser.resolveOperands(operands, funcType.getInputs(), operandsloc,
                               result.operands))
      return mlir::failure();

    result.addTypes(funcType.getResults());
    return mlir::success();
  }

  if (parser.resolveOperands(operands, type, result.operands))
    return mlir::failure();
  result.addTypes(type);
  return mlir::success();
}

static void printBinaryOp(mlir::OpAsmPrinter &printer, mlir::Operation *op) {
  printer << " " << op->getOperands();
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : ";

  mlir::Type resultType = *op->result_type_begin();
  if (llvm::all_of(op->getOperandTypes(),
                   [=](mlir::Type type) { return type == resultType; })) {
    printer << resultType;
    return;
  }

  printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}

//===----------------------------------------------------------------------===//
// AddOp
void mlir::toy::AddOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &state, mlir::Value lhs,
                             mlir::Value rhs) {
  state.addTypes(mlir::UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

//===----------------------------------------------------------------------===//
// MulOp
void mlir::toy::MulOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &state, mlir::Value lhs,
                             mlir::Value rhs) {
  state.addTypes(mlir::UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

//===----------------------------------------------------------------------===//
// ReturnOp

static mlir::LogicalResult verify(mlir::toy::ReturnOp op) {
  auto function = mlir::cast<mlir::FuncOp>(op->getParentOp());

  if (op.getNumOperands() > 1)
    return op.emitOpError() << "expects at most 1 return operand";

  const auto &results = function.getType().getResults();
  if (op.getNumOperands() != results.size())
    return op.emitOpError()
           << "does not return the same number of values ("
           << op.getNumOperands() << ") as the enclosing function ("
           << results.size() << ")";

  if (!op.hasOperand())
    return mlir::success();

  auto inputType = *op.operand_type_begin();
  auto resultType = results.front();

  if (inputType == resultType || inputType.isa<mlir::UnrankedTensorType>() ||
      resultType.isa<mlir::UnrankedTensorType>())
    return mlir::success();

  return op.emitError() << "type of return operand (" << inputType
                        << ") doesn't match function result type ("
                        << resultType << ")";
}

#define GET_OP_CLASSES
#include "toy/Ops.cpp.inc"
