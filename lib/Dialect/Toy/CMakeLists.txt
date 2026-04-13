add_mlir_dialect_library(RYUKIToyDialect
  ToyDialect.cpp
  ToyOps.cpp

  DEPENDS
    RYUKIToyOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    MLIRInferTypeOpInterface
)
