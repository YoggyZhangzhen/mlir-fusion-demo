; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

define { ptr, ptr, i64, [2 x i64], [2 x i64] } @matmul_bias(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, ptr %14, ptr %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20) {
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %14, 0
  %23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, ptr %15, 1
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, i64 %16, 2
  %25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, i64 %17, 3, 0
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, i64 %19, 4, 0
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, i64 %18, 3, 1
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, i64 %20, 4, 1
  %29 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %7, 0
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, ptr %8, 1
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, i64 %9, 2
  %32 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, i64 %10, 3, 0
  %33 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, i64 %12, 4, 0
  %34 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, i64 %11, 3, 1
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, i64 %13, 4, 1
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %0, 0
  %37 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, ptr %1, 1
  %38 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, i64 %2, 2
  %39 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, i64 %3, 3, 0
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, i64 %5, 4, 0
  %41 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, i64 %4, 3, 1
  %42 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, i64 %6, 4, 1
  %43 = call ptr @malloc(i64 192)
  %44 = ptrtoint ptr %43 to i64
  %45 = add i64 %44, 63
  %46 = urem i64 %45, 64
  %47 = sub i64 %45, %46
  %48 = inttoptr i64 %47 to ptr
  %49 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %43, 0
  %50 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %49, ptr %48, 1
  %51 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %50, i64 0, 2
  %52 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %51, i64 4, 3, 0
  %53 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %52, i64 8, 3, 1
  %54 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, i64 8, 4, 0
  %55 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %54, i64 1, 4, 1
  br label %56

56:                                               ; preds = %69, %21
  %57 = phi i64 [ %70, %69 ], [ 0, %21 ]
  %58 = icmp slt i64 %57, 4
  br i1 %58, label %59, label %71

59:                                               ; preds = %56
  br label %60

60:                                               ; preds = %63, %59
  %61 = phi i64 [ %68, %63 ], [ 0, %59 ]
  %62 = icmp slt i64 %61, 8
  br i1 %62, label %63, label %69

63:                                               ; preds = %60
  %64 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 1
  %65 = mul nuw nsw i64 %57, 8
  %66 = add nuw nsw i64 %65, %61
  %67 = getelementptr inbounds nuw float, ptr %64, i64 %66
  store float 0.000000e+00, ptr %67, align 4
  %68 = add i64 %61, 1
  br label %60

69:                                               ; preds = %60
  %70 = add i64 %57, 1
  br label %56

71:                                               ; preds = %56
  br label %72

72:                                               ; preds = %118, %71
  %73 = phi i64 [ %119, %118 ], [ 0, %71 ]
  %74 = icmp slt i64 %73, 4
  br i1 %74, label %75, label %120

75:                                               ; preds = %72
  br label %76

76:                                               ; preds = %116, %75
  %77 = phi i64 [ %117, %116 ], [ 0, %75 ]
  %78 = icmp slt i64 %77, 8
  br i1 %78, label %79, label %118

79:                                               ; preds = %76
  br label %80

80:                                               ; preds = %83, %79
  %81 = phi i64 [ %115, %83 ], [ 0, %79 ]
  %82 = icmp slt i64 %81, 16
  br i1 %82, label %83, label %116

83:                                               ; preds = %80
  %84 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, 1
  %85 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, 2
  %86 = getelementptr float, ptr %84, i64 %85
  %87 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, 4, 0
  %88 = mul nuw nsw i64 %73, %87
  %89 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, 4, 1
  %90 = mul nuw nsw i64 %81, %89
  %91 = add nuw nsw i64 %88, %90
  %92 = getelementptr inbounds nuw float, ptr %86, i64 %91
  %93 = load float, ptr %92, align 4
  %94 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 1
  %95 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 2
  %96 = getelementptr float, ptr %94, i64 %95
  %97 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 4, 0
  %98 = mul nuw nsw i64 %81, %97
  %99 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 4, 1
  %100 = mul nuw nsw i64 %77, %99
  %101 = add nuw nsw i64 %98, %100
  %102 = getelementptr inbounds nuw float, ptr %96, i64 %101
  %103 = load float, ptr %102, align 4
  %104 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 1
  %105 = mul nuw nsw i64 %73, 8
  %106 = add nuw nsw i64 %105, %77
  %107 = getelementptr inbounds nuw float, ptr %104, i64 %106
  %108 = load float, ptr %107, align 4
  %109 = fmul float %93, %103
  %110 = fadd float %108, %109
  %111 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 1
  %112 = mul nuw nsw i64 %73, 8
  %113 = add nuw nsw i64 %112, %77
  %114 = getelementptr inbounds nuw float, ptr %111, i64 %113
  store float %110, ptr %114, align 4
  %115 = add i64 %81, 1
  br label %80

116:                                              ; preds = %80
  %117 = add i64 %77, 1
  br label %76

118:                                              ; preds = %76
  %119 = add i64 %73, 1
  br label %72

120:                                              ; preds = %72
  br label %121

121:                                              ; preds = %150, %120
  %122 = phi i64 [ %151, %150 ], [ 0, %120 ]
  %123 = icmp slt i64 %122, 4
  br i1 %123, label %124, label %152

124:                                              ; preds = %121
  br label %125

125:                                              ; preds = %128, %124
  %126 = phi i64 [ %149, %128 ], [ 0, %124 ]
  %127 = icmp slt i64 %126, 8
  br i1 %127, label %128, label %150

128:                                              ; preds = %125
  %129 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 1
  %130 = mul nuw nsw i64 %122, 8
  %131 = add nuw nsw i64 %130, %126
  %132 = getelementptr inbounds nuw float, ptr %129, i64 %131
  %133 = load float, ptr %132, align 4
  %134 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 1
  %135 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 2
  %136 = getelementptr float, ptr %134, i64 %135
  %137 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 4, 0
  %138 = mul nuw nsw i64 %122, %137
  %139 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 4, 1
  %140 = mul nuw nsw i64 %126, %139
  %141 = add nuw nsw i64 %138, %140
  %142 = getelementptr inbounds nuw float, ptr %136, i64 %141
  %143 = load float, ptr %142, align 4
  %144 = fadd float %133, %143
  %145 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 1
  %146 = mul nuw nsw i64 %122, 8
  %147 = add nuw nsw i64 %146, %126
  %148 = getelementptr inbounds nuw float, ptr %145, i64 %147
  store float %144, ptr %148, align 4
  %149 = add i64 %126, 1
  br label %125

150:                                              ; preds = %125
  %151 = add i64 %122, 1
  br label %121

152:                                              ; preds = %121
  ret { ptr, ptr, i64, [2 x i64], [2 x i64] } %55
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
