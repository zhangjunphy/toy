//===- toyc.cpp - The Toy Compiler ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the entry point for the Toy compiler.
//
//===----------------------------------------------------------------------===//

#include "parser/Parser.h"
#include "toy/Dialect.h"
#include "toy/LoweringPass.h"
#include "toy/MLIRGen.h"
#include "toy/ShapeInferencePass.h"

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace toy;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));
namespace {
enum Action { None, DumpAST, DumpMLIR, DumpAffine, DumpLLVM };
enum InputType { Toy, MLIR };
} // namespace

static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(DumpAffine, "affine", "output the affine dump")),
    cl::values(clEnumValN(DumpLLVM, "llvm", "output the llvm ir dump")));
static cl::opt<enum InputType> inputType(
    "x", cl::init(Toy), cl::desc("Input file format"),
    cl::values(clEnumValN(Toy, "toy", "input file in Toy format.")),
    cl::values(clEnumValN(MLIR, "mlir", "input file in MLIR format")));

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));

/// Returns a Toy AST resulting from parsing the file or a nullptr on error.
std::unique_ptr<toy::ModuleAST> parseInputFile(llvm::StringRef filename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }
  auto buffer = fileOrErr.get()->getBuffer();
  LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
  Parser parser(lexer);
  return parser.parseModule();
}

mlir::OwningOpRef<mlir::ModuleOp> parseMLIRModule(mlir::MLIRContext *context,
                                                  llvm::StringRef filename) {
  // Parse from toy ast or mlir asm file
  if (inputType != InputType::MLIR &&
      !llvm::StringRef(inputFilename).endswith(".mlir")) {
    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST)
      return nullptr;
    return mlirGen(*context, *moduleAST);
  } else {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code ec = fileOrErr.getError()) {
      llvm::errs() << "Cannot open file: " << ec.message() << "\n";
      return nullptr;
    }

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    return mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, context);
  }
}

int dumpAST() {
  if (inputType == InputType::MLIR) {
    llvm::errs() << "Cannot dump mlir file into toy ast\n";
    return 5;
  }

  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST)
    return 1;

  dump(*moduleAST);
  return 0;
}

int dumpMLIR() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::toy::ToyDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseMLIRModule(&context, inputFilename);
  if (!module) {
    llvm::errs() << "Error parsing input file " + inputFilename + "\n";
    return 2;
  }

  mlir::PassManager pm(&context);
  mlir::applyPassManagerCLOptions(pm);
  pm.addPass(mlir::createInlinerPass());
  pm.addNestedPass<mlir::toy::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::toy::FuncOp>(mlir::toy::createShapeInferencePass());
  if (mlir::failed(pm.run(*module))) {
    return 4;
  }

  module->dump();
  return 0;
}

int dumpAffine() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::toy::ToyDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseMLIRModule(&context, inputFilename);
  if (!module) {
    llvm::errs() << "Error parsing input file " + inputFilename + "\n";
    return 2;
  }

  mlir::PassManager pm(&context);
  mlir::applyPassManagerCLOptions(pm);
  pm.addPass(mlir::createInlinerPass());
  pm.addNestedPass<mlir::toy::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::toy::FuncOp>(mlir::toy::createShapeInferencePass());
  pm.addNestedPass<mlir::toy::FuncOp>(mlir::createCSEPass());

  pm.addPass(mlir::toy::createLoweringPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
  if (enableOpt) {
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createLoopFusionPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::createAffineScalarReplacementPass());
  }
  if (mlir::failed(pm.run(*module))) {
    return 4;
  }

  module->dump();
  return 0;
}

int dumpLLVM() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::toy::ToyDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseMLIRModule(&context, inputFilename);
  if (!module) {
    llvm::errs() << "Error parsing input file " + inputFilename + "\n";
    return 2;
  }

  mlir::PassManager pm(&context);
  mlir::applyPassManagerCLOptions(pm);
  pm.addPass(mlir::createInlinerPass());
  pm.addNestedPass<mlir::toy::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::toy::FuncOp>(mlir::toy::createShapeInferencePass());
  pm.addNestedPass<mlir::toy::FuncOp>(mlir::createCSEPass());

  pm.addPass(mlir::toy::createLoweringPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
  if (enableOpt) {
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createLoopFusionPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::createAffineScalarReplacementPass());
  }
  pm.addPass(mlir::toy::createLoweringToLLVMPass());
  if (mlir::failed(pm.run(*module))) {
    return 4;
  }
  //module->dump();

  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(module->getOperation(), llvmContext);
  if (!llvmModule) {
    return 4;
  }
  llvmModule->dump();

  return 0;
}

int main(int argc, char **argv) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

  switch (emitAction) {
  case Action::DumpAST:
    return dumpAST();
  case Action::DumpMLIR:
    return dumpMLIR();
  case Action::DumpAffine:
    return dumpAffine();
  case Action::DumpLLVM:
    return dumpLLVM();
  default:
    llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  }

  return 0;
}
