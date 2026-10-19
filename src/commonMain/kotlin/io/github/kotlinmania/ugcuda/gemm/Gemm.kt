// port-lint: source src/gemm.rs
package io.github.kotlinmania.ugcuda.gemm

import io.github.kotlinmania.cudarc.cublas.result.CublasComputeType
import io.github.kotlinmania.cudarc.cublas.result.CublasGemmAlgo
import io.github.kotlinmania.cudarc.cublas.result.CublasOperation
import io.github.kotlinmania.cudarc.cublas.result.CudaDataType
import io.github.kotlinmania.cudarc.cublas.result.gemmStridedBatchedEx
import io.github.kotlinmania.cudarc.cublas.safe.CudaBlas
import io.github.kotlinmania.cudarc.cublas.safe.GemmConfig
import io.github.kotlinmania.cudarc.cublas.safe.StridedBatchedConfig
import io.github.kotlinmania.cudarc.driver.safe.CudaSlice
import io.github.kotlinmania.cudarc.driver.safe.CudaStream
import io.github.kotlinmania.cudarc.driver.safe.CudaView
import io.github.kotlinmania.half.BF16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.ug.Layout
import io.github.kotlinmania.ug.UgError
import io.github.kotlinmania.ugcuda.runtime.Slice
import io.github.kotlinmania.ugcuda.runtime.SliceInner
import kotlin.concurrent.AtomicBoolean

internal fun <T> gemmConfig(
    alpha: T,
    beta: T,
    bmnk: List<Int>,
    lhsL: Layout,
    rhsL: Layout,
): StridedBatchedConfig<T> {
    val b = bmnk[0]
    val m = bmnk[1]
    val n = bmnk[2]
    val k = bmnk[3]

    val lhsStride = lhsL.strides
    val rhsStride = rhsL.strides
    val rhsM1 = rhsStride[rhsStride.size - 1]
    val rhsM2 = rhsStride[rhsStride.size - 2]
    val lhsM1 = lhsStride[lhsStride.size - 1]
    val lhsM2 = lhsStride[lhsStride.size - 2]

    val (lda, transa) = if ((rhsM1 == 1 || n == 1) && (rhsM2 == n || k == 1)) {
        Pair(n, CublasOperation.CUBLAS_OP_N)
    } else if ((rhsM1 == k || n == 1) && (rhsM2 == 1 || k == 1)) {
        Pair(k, CublasOperation.CUBLAS_OP_T)
    } else {
        throw UgError.Msg("non contiguous matmul, lhs: $lhsL, rhs: $rhsL")
    }

    val (ldb, transb) = if ((lhsM1 == 1 || k == 1) && (lhsM2 == k || m == 1)) {
        Pair(k, CublasOperation.CUBLAS_OP_N)
    } else if ((lhsM1 == m || k == 1) && (lhsM2 == 1 || m == 1)) {
        Pair(m, CublasOperation.CUBLAS_OP_T)
    } else {
        throw UgError.Msg("non contiguous matmul, lhs: $lhsL, rhs: $rhsL")
    }

    val gemm = GemmConfig(
        transa = transa,
        transb = transb,
        m = n,
        n = m,
        k = k,
        alpha = alpha,
        lda = lda,
        ldb = ldb,
        beta = beta,
        ldc = n,
    )

    val lhsLeading = lhsStride.subList(0, lhsStride.size - 2)
    val strideB: Long = when {
        lhsLeading.size == 2 && lhsLeading[0] == lhsLeading[1] * lhsL.dims[1] -> lhsLeading[1].toLong()
        lhsLeading.size == 2 && lhsL.dims[0] == 1 -> lhsLeading[1].toLong()
        lhsLeading.size == 2 && lhsL.dims[1] == 1 -> lhsLeading[0].toLong()
        lhsLeading.size == 1 -> lhsLeading[0].toLong()
        lhsLeading.isEmpty() -> (m * k).toLong()
        else -> throw UgError.Msg("non contiguous matmul, lhs: $lhsL, rhs: $rhsL")
    }

    val rhsLeading = rhsStride.subList(0, rhsStride.size - 2)
    val strideA: Long = when {
        rhsLeading.size == 2 && rhsLeading[0] == rhsLeading[1] * rhsL.dims[1] -> rhsLeading[1].toLong()
        rhsLeading.size == 2 && rhsL.dims[0] == 1 -> rhsLeading[1].toLong()
        rhsLeading.size == 2 && rhsL.dims[1] == 1 -> rhsLeading[0].toLong()
        rhsLeading.size == 1 -> rhsLeading[0].toLong()
        rhsLeading.isEmpty() -> (n * k).toLong()
        else -> throw UgError.Msg("non contiguous matmul, lhs: $lhsL, rhs: $rhsL")
    }

    return StridedBatchedConfig(
        gemm = gemm,
        batchSize = b,
        strideA = strideA,
        strideB = strideB,
        strideC = (m * n).toLong(),
    )
}

private val MM_F16_REDUCED_PRECISION: AtomicBoolean = AtomicBoolean(false)
private val MM_BF16_REDUCED_PRECISION: AtomicBoolean = AtomicBoolean(false)
private val MM_F32_REDUCED_PRECISION: AtomicBoolean = AtomicBoolean(false)

public fun gemmReducedPrecisionF32(): Boolean =
    MM_F32_REDUCED_PRECISION.value

public fun setGemmReducedPrecisionF32(b: Boolean) {
    MM_F32_REDUCED_PRECISION.value = b
}

public fun gemmReducedPrecisionF16(): Boolean =
    MM_F16_REDUCED_PRECISION.value

public fun setGemmReducedPrecisionF16(b: Boolean) {
    MM_F16_REDUCED_PRECISION.value = b
}

public fun gemmReducedPrecisionBf16(): Boolean =
    MM_BF16_REDUCED_PRECISION.value

public fun setGemmReducedPrecisionBf16(b: Boolean) {
    MM_BF16_REDUCED_PRECISION.value = b
}

internal fun gemmStridedBatchedF32(
    stream: CudaStream,
    cublas: CudaBlas,
    cfg: StridedBatchedConfig<Float>,
    a: CudaView<Float>,
    b: CudaView<Float>,
    c: CudaSlice<Float>,
): kotlin.Result<Unit> {
    val computeType = if (gemmReducedPrecisionF32()) {
        CublasComputeType.CUBLAS_COMPUTE_32F_FAST_TF32
    } else {
        CublasComputeType.CUBLAS_COMPUTE_32F
    }
    val aPtr = a.devicePtr(stream).first
    val bPtr = b.devicePtr(stream).first
    val cPtr = c.devicePtr(stream).first

    return gemmStridedBatchedEx(
        handle = cublas.handle,
        transa = cfg.gemm.transa,
        transb = cfg.gemm.transb,
        m = cfg.gemm.m,
        n = cfg.gemm.n,
        k = cfg.gemm.k,
        alpha = cfg.gemm.alpha.toBits().toLong(),
        a = aPtr,
        aType = CudaDataType.CUDA_R_32F,
        lda = cfg.gemm.lda,
        strideA = cfg.strideA,
        b = bPtr,
        bType = CudaDataType.CUDA_R_32F,
        ldb = cfg.gemm.ldb,
        strideB = cfg.strideB,
        beta = cfg.gemm.beta.toBits().toLong(),
        c = cPtr,
        cType = CudaDataType.CUDA_R_32F,
        ldc = cfg.gemm.ldc,
        strideC = cfg.strideC,
        batchCount = cfg.batchSize,
        computeType = computeType,
        algo = CublasGemmAlgo.CUBLAS_GEMM_DEFAULT,
    )
}

internal fun gemmStridedBatchedF16(
    stream: CudaStream,
    cublas: CudaBlas,
    cfg: StridedBatchedConfig<F16>,
    a: CudaView<F16>,
    b: CudaView<F16>,
    c: CudaSlice<F16>,
): kotlin.Result<Unit> {
    val computeType = if (gemmReducedPrecisionF16()) {
        CublasComputeType.CUBLAS_COMPUTE_16F
    } else {
        CublasComputeType.CUBLAS_COMPUTE_32F
    }
    val alphaBits = if (gemmReducedPrecisionF16()) {
        cfg.gemm.alpha.toBits().toLong()
    } else {
        cfg.gemm.alpha.toFloat().toBits().toLong()
    }
    val betaBits = if (gemmReducedPrecisionF16()) {
        cfg.gemm.beta.toBits().toLong()
    } else {
        cfg.gemm.beta.toFloat().toBits().toLong()
    }

    val aPtr = a.devicePtr(stream).first
    val bPtr = b.devicePtr(stream).first
    val cPtr = c.devicePtr(stream).first

    return gemmStridedBatchedEx(
        handle = cublas.handle,
        transa = cfg.gemm.transa,
        transb = cfg.gemm.transb,
        m = cfg.gemm.m,
        n = cfg.gemm.n,
        k = cfg.gemm.k,
        alpha = alphaBits,
        a = aPtr,
        aType = CudaDataType.CUDA_R_16F,
        lda = cfg.gemm.lda,
        strideA = cfg.strideA,
        b = bPtr,
        bType = CudaDataType.CUDA_R_16F,
        ldb = cfg.gemm.ldb,
        strideB = cfg.strideB,
        beta = betaBits,
        c = cPtr,
        cType = CudaDataType.CUDA_R_16F,
        ldc = cfg.gemm.ldc,
        strideC = cfg.strideC,
        batchCount = cfg.batchSize,
        computeType = computeType,
        algo = CublasGemmAlgo.CUBLAS_GEMM_DEFAULT,
    )
}

internal fun gemmStridedBatchedBf16(
    stream: CudaStream,
    cublas: CudaBlas,
    cfg: StridedBatchedConfig<BF16>,
    a: CudaView<BF16>,
    b: CudaView<BF16>,
    c: CudaSlice<BF16>,
): kotlin.Result<Unit> {
    val computeType = if (gemmReducedPrecisionBf16()) {
        CublasComputeType.CUBLAS_COMPUTE_32F_FAST_16BF
    } else {
        CublasComputeType.CUBLAS_COMPUTE_32F
    }
    val alphaBits = cfg.gemm.alpha.toFloat().toBits().toLong()
    val betaBits = cfg.gemm.beta.toFloat().toBits().toLong()

    val aPtr = a.devicePtr(stream).first
    val bPtr = b.devicePtr(stream).first
    val cPtr = c.devicePtr(stream).first

    return gemmStridedBatchedEx(
        handle = cublas.handle,
        transa = cfg.gemm.transa,
        transb = cfg.gemm.transb,
        m = cfg.gemm.m,
        n = cfg.gemm.n,
        k = cfg.gemm.k,
        alpha = alphaBits,
        a = aPtr,
        aType = CudaDataType.CUDA_R_16BF,
        lda = cfg.gemm.lda,
        strideA = cfg.strideA,
        b = bPtr,
        bType = CudaDataType.CUDA_R_16BF,
        ldb = cfg.gemm.ldb,
        strideB = cfg.strideB,
        beta = betaBits,
        c = cPtr,
        cType = CudaDataType.CUDA_R_16BF,
        ldc = cfg.gemm.ldc,
        strideC = cfg.strideC,
        batchCount = cfg.batchSize,
        computeType = computeType,
        algo = CublasGemmAlgo.CUBLAS_GEMM_DEFAULT,
    )
}

internal fun matmul(
    blas: CudaBlas,
    dst: Slice,
    lhs: Slice,
    rhs: Slice,
    bmnk: List<Int>,
    lhsL: Layout,
    rhsL: Layout,
) {
    val dstDt = dst.dtype()
    val lhsDt = lhs.dtype()
    val rhsDt = rhs.dtype()
    val dstInner = dst.inner
    val lhsInner = lhs.inner
    val rhsInner = rhs.inner

    when {
        dstInner is SliceInner.Bf16Val && lhsInner is SliceInner.Bf16Val && rhsInner is SliceInner.Bf16Val -> {
            val stream = lhsInner.slice.stream
            val lhsView = lhsInner.slice.slice(lhsL.offset until lhsInner.slice.len)
            val rhsView = rhsInner.slice.slice(rhsL.offset until rhsInner.slice.len)
            val cfg = gemmConfig(BF16.ONE, BF16.ZERO, bmnk, lhsL, rhsL)
            gemmStridedBatchedBf16(stream, blas, cfg, rhsView, lhsView, dstInner.slice).getOrThrow()
        }
        dstInner is SliceInner.F16Val && lhsInner is SliceInner.F16Val && rhsInner is SliceInner.F16Val -> {
            val stream = lhsInner.slice.stream
            val lhsView = lhsInner.slice.slice(lhsL.offset until lhsInner.slice.len)
            val rhsView = rhsInner.slice.slice(rhsL.offset until rhsInner.slice.len)
            val cfg = gemmConfig(F16.ONE, F16.ZERO, bmnk, lhsL, rhsL)
            gemmStridedBatchedF16(stream, blas, cfg, rhsView, lhsView, dstInner.slice).getOrThrow()
        }
        dstInner is SliceInner.F32 && lhsInner is SliceInner.F32 && rhsInner is SliceInner.F32 -> {
            val stream = lhsInner.slice.stream
            val lhsView = lhsInner.slice.slice(lhsL.offset until lhsInner.slice.len)
            val rhsView = rhsInner.slice.slice(rhsL.offset until rhsInner.slice.len)
            val cfg = gemmConfig(1.0f, 0.0f, bmnk, lhsL, rhsL)
            gemmStridedBatchedF32(stream, blas, cfg, rhsView, lhsView, dstInner.slice).getOrThrow()
        }
        else -> {
            throw UgError.Msg("incorrect dtypes for matmul, dst: $dstDt, lhs: $lhsDt, rhs: $rhsDt")
        }
    }
}
