; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define { ptr, ptr, i64, [2 x i64], [2 x i64] } @matmul_bias(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, ptr %14, ptr %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20) {
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %7, 0
  %23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, ptr %8, 1
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, i64 %9, 2
  %25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, i64 %10, 3, 0
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, i64 %12, 4, 0
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, i64 %11, 3, 1
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, i64 %13, 4, 1
  %29 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %0, 0
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, ptr %1, 1
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, i64 %2, 2
  %32 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, i64 %3, 3, 0
  %33 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, i64 %5, 4, 0
  %34 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, i64 %4, 3, 1
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, i64 %6, 4, 1
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %14, 0
  %37 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, ptr %15, 1
  %38 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, i64 %16, 2
  %39 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, i64 %17, 3, 0
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, i64 %19, 4, 0
  %41 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, i64 %18, 3, 1
  %42 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, i64 %20, 4, 1
  br label %43

43:                                               ; preds = %99, %21
  %44 = phi i64 [ %100, %99 ], [ 0, %21 ]
  %45 = icmp slt i64 %44, 4
  br i1 %45, label %46, label %101

46:                                               ; preds = %43
  br label %47

47:                                               ; preds = %97, %46
  %48 = phi i64 [ %98, %97 ], [ 0, %46 ]
  %49 = icmp slt i64 %48, 8
  br i1 %49, label %50, label %99

50:                                               ; preds = %47
  br label %51

51:                                               ; preds = %54, %50
  %52 = phi i64 [ %96, %54 ], [ 0, %50 ]
  %53 = icmp slt i64 %52, 16
  br i1 %53, label %54, label %97

54:                                               ; preds = %51
  %55 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 1
  %56 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 2
  %57 = getelementptr float, ptr %55, i64 %56
  %58 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 4, 0
  %59 = mul nuw nsw i64 %44, %58
  %60 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 4, 1
  %61 = mul nuw nsw i64 %52, %60
  %62 = add nuw nsw i64 %59, %61
  %63 = getelementptr inbounds nuw float, ptr %57, i64 %62
  %64 = load float, ptr %63, align 4
  %65 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 1
  %66 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 2
  %67 = getelementptr float, ptr %65, i64 %66
  %68 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 4, 0
  %69 = mul nuw nsw i64 %52, %68
  %70 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 4, 1
  %71 = mul nuw nsw i64 %48, %70
  %72 = add nuw nsw i64 %69, %71
  %73 = getelementptr inbounds nuw float, ptr %67, i64 %72
  %74 = load float, ptr %73, align 4
  %75 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, 1
  %76 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, 2
  %77 = getelementptr float, ptr %75, i64 %76
  %78 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, 4, 0
  %79 = mul nuw nsw i64 %44, %78
  %80 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, 4, 1
  %81 = mul nuw nsw i64 %48, %80
  %82 = add nuw nsw i64 %79, %81
  %83 = getelementptr inbounds nuw float, ptr %77, i64 %82
  %84 = load float, ptr %83, align 4
  %85 = fmul float %64, %74
  %86 = fadd float %84, %85
  %87 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, 1
  %88 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, 2
  %89 = getelementptr float, ptr %87, i64 %88
  %90 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, 4, 0
  %91 = mul nuw nsw i64 %44, %90
  %92 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, 4, 1
  %93 = mul nuw nsw i64 %48, %92
  %94 = add nuw nsw i64 %91, %93
  %95 = getelementptr inbounds nuw float, ptr %89, i64 %94
  store float %86, ptr %95, align 4
  %96 = add i64 %52, 1
  br label %51

97:                                               ; preds = %51
  %98 = add i64 %48, 1
  br label %47

99:                                               ; preds = %47
  %100 = add i64 %44, 1
  br label %43

101:                                              ; preds = %43
  ret { ptr, ptr, i64, [2 x i64], [2 x i64] } %42
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
