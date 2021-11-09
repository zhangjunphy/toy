#include "toy/MLIRGen.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "toy/AST.h"
#include "toy/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "toy/Lexer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <functional>
#include <numeric>

namespace {

class MLIRGenImpl {
public:
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

  mlir::ModuleOp mlirGen(toy::ModuleAST &moduleAst) {
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    for (toy::FunctionAST &F : moduleAst) {
      auto func = mlirGen(F);
      if (!func)
        return nullptr;
      theModule.push_back(func);
    }

    return theModule;
  }

private:
  mlir::ModuleOp theModule;
  mlir::OpBuilder builder;
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;

  mlir::Location loc(toy::Location loc) {
    return mlir::FileLineColLoc::get(builder.getIdentifier(*loc.file), loc.line,
                                     loc.col);
  }

  mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
    if (symbolTable.count(var))
      return mlir::failure();
    symbolTable.insert(var, value);
    return mlir::success();
  }

  mlir::FuncOp mlirGen(toy::PrototypeAST &proto) {
    auto location = loc(proto.loc());
    llvm::SmallVector<mlir::Type, 4> arg_types(proto.getArgs().size(),
                                               getType(toy::VarType{}));
    auto func_type = builder.getFunctionType(arg_types, llvm::None);
    return mlir::FuncOp::create(location, proto.getName(), func_type);
  }

  mlir::FuncOp mlirGen(toy::FunctionAST &funcAST) {
    llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> var_scope(
        symbolTable);

    mlir::FuncOp function(mlirGen(*funcAST.getProto()));
    if (!function)
      return nullptr;

    auto &entryBlock = *function.addEntryBlock();
    auto protoArgs = funcAST.getProto()->getArgs();

    for (const auto nameValue :
         llvm::zip(protoArgs, entryBlock.getArguments())) {
      if (mlir::failed(declare(std::get<0>(nameValue)->getName(),
                               std::get<1>(nameValue))))
        return nullptr;
    }

    builder.setInsertionPointToStart(&entryBlock);

    if (mlir::failed(mlirGen(*funcAST.getBody()))) {
      function.erase();
      return nullptr;
    }

    mlir::toy::ReturnOp returnOp;
    if (!entryBlock.empty())
      returnOp = mlir::dyn_cast<mlir::toy::ReturnOp>(entryBlock.back());
    if (!returnOp) {
      builder.create<mlir::toy::ReturnOp>(loc(funcAST.getProto()->loc()));
    } else if (returnOp.hasOperand()) {
      function.setType(builder.getFunctionType(function.getType().getInputs(),
                                               getType(toy::VarType{})));
    }

    return function;
  }

  mlir::LogicalResult mlirGen(toy::ExprASTList &blockAST) {
    llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> var_scope(
        symbolTable);
    for (auto &expr : blockAST) {
      // statements should only appear in top level of a block
      if (auto *vardecl = mlir::dyn_cast<toy::VarDeclExprAST>(expr.get())) {
        if (!mlirGen(*vardecl))
          return mlir::failure();
        continue;
      }
      if (auto *ret = mlir::dyn_cast<toy::ReturnExprAST>(expr.get()))
        return mlirGen(*ret);
      if (auto *print = mlir::dyn_cast<toy::PrintExprAST>(expr.get())) {
        if (mlir::failed(mlirGen(*print)))
          return mlir::failure();
        continue;
      }

      // handle real expressions which could be nested
      if (!mlirGen(*expr))
        return mlir::failure();
    }

    return mlir::success();
  }

  mlir::Value mlirGen(toy::VarDeclExprAST &vardecl) {
    auto init = vardecl.getInitVal();
    if (!init) {
      mlir::emitError(loc(vardecl.loc()), "missing initial value");
      return nullptr;
    }

    mlir::Value value = mlirGen(*init);
    if (!value)
      return nullptr;

    if (!vardecl.getType().shape.empty()) {
      value = builder.create<mlir::toy::ReshapeOp>(
          loc(vardecl.loc()), getType(vardecl.getType()), value);
    }

    if (mlir::failed(declare(vardecl.getName(), value)))
      return nullptr;
    return value;
  }

  mlir::LogicalResult mlirGen(toy::ReturnExprAST &ret) {
    auto location = loc(ret.loc());

    mlir::Value expr = nullptr;
    if (ret.getExpr().hasValue()) {
      if (!(expr = mlirGen(*ret.getExpr().getValue())))
        return mlir::failure();
    }

    builder.create<mlir::toy::ReturnOp>(location,
                                        expr ? llvm::makeArrayRef(expr)
                                             : llvm::ArrayRef<mlir::Value>());
    return mlir::success();
  }

  mlir::LogicalResult mlirGen(toy::PrintExprAST &print) {
    auto location = loc(print.loc());

    if (mlir::Value expr = mlirGen(*print.getArg())) {
      builder.create<mlir::toy::PrintOp>(location, expr);
      return mlir::success();
    }

    return mlir::failure();
  }

  mlir::Value mlirGen(toy::ExprAST &expr) {
    switch (expr.getKind()) {
    case toy::ExprAST::Expr_BinOp:
      return mlirGen(mlir::cast<toy::BinaryExprAST>(expr));
    case toy::ExprAST::Expr_Var:
      return mlirGen(mlir::cast<toy::VariableExprAST>(expr));
    case toy::ExprAST::Expr_Literal:
      return mlirGen(mlir::cast<toy::LiteralExprAST>(expr));
    case toy::ExprAST::Expr_Num:
      return mlirGen(mlir::cast<toy::NumberExprAST>(expr));
    case toy::ExprAST::Expr_Call:
      return mlirGen(mlir::cast<toy::CallExprAST>(expr));
    default:
      mlir::emitError(loc(expr.loc()))
          << "Unhandled expr kind '" << mlir::Twine(expr.getKind()) << "'";
      return nullptr;
    }
  }

  mlir::Value mlirGen(toy::BinaryExprAST &bin) {
    mlir::Value lhs = mlirGen(*bin.getLHS());
    if (!lhs)
      return nullptr;
    mlir::Value rhs = mlirGen(*bin.getRHS());
    if (!rhs)
      return nullptr;

    auto location = loc(bin.loc());

    switch (bin.getOp()) {
    case '+':
      return builder.create<mlir::toy::AddOp>(location, lhs, rhs);
    case '*':
      return builder.create<mlir::toy::MulOp>(location, lhs, rhs);
    }

    mlir::emitError(location, "invalid binary operator '")
        << bin.getOp() << "'";
    return nullptr;
  }

  mlir::Value mlirGen(toy::VariableExprAST &var) {
    if (auto variable = symbolTable.lookup(var.getName()))
      return variable;

    mlir::emitError(loc(var.loc()), "unknown variable: '")
        << var.getName() << "'";
    return nullptr;
  }

  mlir::Value mlirGen(toy::LiteralExprAST &lit) {
    auto type = getType(lit.getDims());

    std::vector<double> data;
    data.reserve(std::accumulate(lit.getDims().begin(), lit.getDims().end(), 1,
                                 std::multiplies<int>()));
    collectData(lit, data);

    mlir::Type elementType = builder.getF64Type();
    auto dataType = mlir::RankedTensorType::get(lit.getDims(), elementType);
    auto dataAttribute =
        mlir::DenseElementsAttr::get(dataType, llvm::makeArrayRef(data));

    return builder.create<mlir::toy::ConstantOp>(loc(lit.loc()), type,
                                                 dataAttribute);
  }

  mlir::Value mlirGen(toy::NumberExprAST &num) {
    std::vector<double> data = {num.getValue()};

    mlir::Type elementType = builder.getF64Type();
    auto dataType = mlir::RankedTensorType::get({1}, elementType);
    auto dataAttribute =
        mlir::DenseElementsAttr::get(dataType, llvm::makeArrayRef(data));

    return builder.create<mlir::toy::ConstantOp>(loc(num.loc()), dataType,
                                                 dataAttribute);
  }

  mlir::Value mlirGen(toy::CallExprAST &call) {
    llvm::StringRef callee = call.getCallee();
    mlir::Location location = loc(call.loc());

    mlir::SmallVector<mlir::Value, 4> operands;
    for (auto &expr : call.getArgs()) {
      auto arg = mlirGen(*expr);
      if (!arg)
        return nullptr;
      operands.push_back(arg);
    }

    if (callee == "transpose") {
      if (call.getArgs().size() != 1) {
        mlir::emitError(location,
                        "toy.transpose does not accept multiple arguments");
        return nullptr;
      }

      return builder.create<mlir::toy::TransposeOp>(
          location, mlir::UnrankedTensorType::get(builder.getF64Type()),
          operands[0]);
    }

    // return builder.create<GenericCallOp>(location, call.getArgs());
  }

  void collectData(toy::ExprAST &expr, std::vector<double> &data) {
    if (auto *lit = mlir::dyn_cast<toy::LiteralExprAST>(&expr)) {
      for (auto &value : lit->getValues())
        collectData(*value, data);
      return;
    }

    assert(mlir::isa<toy::NumberExprAST>(expr) && "expected literal or number");
    data.push_back(mlir::cast<toy::NumberExprAST>(expr).getValue());
  }

  mlir::Type getType(mlir::ArrayRef<int64_t> shape) {
    if (shape.empty())
      return mlir::UnrankedTensorType::get(builder.getF64Type());

    return mlir::RankedTensorType::get(shape, builder.getF64Type());
  }

  mlir::Type getType(const toy::VarType &type) { return getType(type.shape); }
};

} // namespace

namespace toy {

mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context,
                              ModuleAST &moduleAST) {
  return MLIRGenImpl(context).mlirGen(moduleAST);
}
} // namespace toy
