add_mlir_library(RYUKITransforms
  MatmulBiasFusion.cpp

  DEPENDS
    RYUKITransformsPassIncGen

  LINK_LIBS PUBLIC
    RYUKIToyDialect
    MLIRPass
    MLIRTransforms
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRArithDialect
    MLIRTensorDialect
    MLIRFuncDialect
    MLIRAffineAnalysis
)
