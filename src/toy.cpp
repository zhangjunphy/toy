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

#include "toy/AST.h"
#include "toy/Dialect.h"
#include "toy/MLIRGen.h"
#include "toy/Parser.h"
#include "toy/Passes.h"

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace toy;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));
namespace {
enum Action { None, DumpAST, DumpMLIR, DumpMLIRAffine };
}
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(DumpMLIRAffine, "affine", "output the affine dump")));

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

int dumpMLIR() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::toy::ToyDialect>();

  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST)
    return 6;
  mlir::OwningModuleRef module = mlirGen(context, *moduleAST);
  if (!module)
    return 1;

  mlir::PassManager pm(&context);
  mlir::applyPassManagerCLOptions(pm);

  bool isLoweringToAffine = emitAction >= Action::DumpMLIRAffine;

  if (enableOpt) {
    mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
    optPM.addPass(mlir::toy::createShapeInferencePass());
    optPM.addPass(mlir::createCanonicalizerPass());
  }

  if (isLoweringToAffine) {
    mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
    optPM.addPass(mlir::toy::createShapeInferencePass());
    optPM.addPass(mlir::toy::createLowerToAffinePass());
    optPM.addPass(mlir::createCanonicalizerPass());

    if (enableOpt) {
      optPM.addPass(mlir::createLoopFusionPass());
      optPM.addPass(mlir::createAffineScalarReplacementPass());
    }
  }

  if (mlir::failed(pm.run(*module)))
    return 4;

  module->dump();
  return 0;
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST)
    return 1;

  switch (emitAction) {
  case Action::DumpAST:
    dump(*moduleAST);
    return 0;
  case Action::DumpMLIR:
  case Action::DumpMLIRAffine:
    return dumpMLIR();
  default:
    llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  }

  return 0;
}
