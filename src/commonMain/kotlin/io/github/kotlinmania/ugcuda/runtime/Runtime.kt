// port-lint: source src/runtime.rs
package io.github.kotlinmania.ugcuda.runtime

import io.github.kotlinmania.cudarc.cublas.safe.CudaBlas
import io.github.kotlinmania.cudarc.driver.safe.CudaContext
import io.github.kotlinmania.cudarc.driver.safe.CudaEvent
import io.github.kotlinmania.cudarc.driver.safe.CudaFunction
import io.github.kotlinmania.cudarc.driver.safe.CudaSlice
import io.github.kotlinmania.cudarc.driver.safe.CudaStream
import io.github.kotlinmania.cudarc.driver.safe.Dim3
import io.github.kotlinmania.cudarc.driver.safe.LaunchArgs
import io.github.kotlinmania.cudarc.driver.safe.LaunchConfig
import io.github.kotlinmania.cudarc.driver.sys.CUdeviceptr
import io.github.kotlinmania.cudarc.nvrtc.safe.CompileOptions
import io.github.kotlinmania.cudarc.nvrtc.safe.Ptx
import io.github.kotlinmania.cudarc.nvrtc.safe.compilePtxWithOpts
import io.github.kotlinmania.half.BF16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.ug.DType
import io.github.kotlinmania.ug.Layout
import io.github.kotlinmania.ug.UgError
import io.github.kotlinmania.ug.lang.ssa.Kernel
import io.github.kotlinmania.ugcuda.codegen.gen
import io.github.kotlinmania.ugcuda.gemm.matmul as gemmMatmul
import kotlin.concurrent.AtomicInt

public interface WithErr<T> {
    public fun w(): T
}

public fun <T> kotlin.Result<T>.w(): T = getOrThrow()

public data class KernelId(
    private val value: Int,
) {
    public fun asUsize(): Int = value

    public companion object {
        private val counter: AtomicInt = AtomicInt(1)

        public fun new(): KernelId =
            KernelId(counter.getAndIncrement())
    }
}

public class Func(
    public val func: CudaFunction,
    public val stream: CudaStream,
    public val cfg: LaunchConfig,
) {
    public companion object {
        public fun new(device: Device, func: CudaFunction, cfg: LaunchConfig): Func =
            Func(func, device.cudarcStream(), cfg)
    }

    public fun builder(): LaunchArgs =
        LaunchArgs(stream, func)

    public fun launchCfg(): LaunchConfig = cfg
}

public sealed class SliceInner {
    public data class F32(val slice: CudaSlice<Float>) : SliceInner()
    public data class F16Val(val slice: CudaSlice<F16>) : SliceInner()
    public data class Bf16Val(val slice: CudaSlice<BF16>) : SliceInner()
    public data class I32(val slice: CudaSlice<Int>) : SliceInner()
    public data class I64(val slice: CudaSlice<Long>) : SliceInner()
}

public interface ToSlice<T> {
    public fun slice(s: Slice): CudaSlice<T>
}

public object ToSliceF32 : ToSlice<Float> {
    override fun slice(s: Slice): CudaSlice<Float> =
        when (val inner = s.inner) {
            is SliceInner.F32 -> inner.slice
            else -> throw UgError.Msg("dtype mismatch, expected ${DType.F32}, got ${s.dtype()}")
        }
}

public object ToSliceF16 : ToSlice<F16> {
    override fun slice(s: Slice): CudaSlice<F16> =
        when (val inner = s.inner) {
            is SliceInner.F16Val -> inner.slice
            else -> throw UgError.Msg("dtype mismatch, expected ${DType.F16}, got ${s.dtype()}")
        }
}

public object ToSliceBf16 : ToSlice<BF16> {
    override fun slice(s: Slice): CudaSlice<BF16> =
        when (val inner = s.inner) {
            is SliceInner.Bf16Val -> inner.slice
            else -> throw UgError.Msg("dtype mismatch, expected ${DType.BF16}, got ${s.dtype()}")
        }
}

public object ToSliceI32 : ToSlice<Int> {
    override fun slice(s: Slice): CudaSlice<Int> =
        when (val inner = s.inner) {
            is SliceInner.I32 -> inner.slice
            else -> throw UgError.Msg("dtype mismatch, expected ${DType.I32}, got ${s.dtype()}")
        }
}

public object ToSliceI64 : ToSlice<Long> {
    override fun slice(s: Slice): CudaSlice<Long> =
        when (val inner = s.inner) {
            is SliceInner.I64 -> inner.slice
            else -> throw UgError.Msg("dtype mismatch, expected ${DType.I64}, got ${s.dtype()}")
        }
}

public class Slice(
    internal var inner: SliceInner,
    override val device: Device,
) : io.github.kotlinmania.ug.Slice {
    override val dtype: DType
        get() = when (inner) {
            is SliceInner.F32 -> DType.F32
            is SliceInner.F16Val -> DType.F16
            is SliceInner.Bf16Val -> DType.BF16
            is SliceInner.I32 -> DType.I32
            is SliceInner.I64 -> DType.I64
        }

    override val len: Int
        get() = when (val s = inner) {
            is SliceInner.F32 -> s.slice.len
            is SliceInner.F16Val -> s.slice.len
            is SliceInner.Bf16Val -> s.slice.len
            is SliceInner.I32 -> s.slice.len
            is SliceInner.I64 -> s.slice.len
        }

    public fun dtype(): DType = dtype

    public fun device(): Device = device

    public fun len(): Int = len

    public fun <D> slice(toSlice: ToSlice<D>): CudaSlice<D> =
        toSlice.slice(this)

    public fun devicePtr(stream: CudaStream): Pair<CUdeviceptr, CudaEvent?> =
        when (val s = inner) {
            is SliceInner.Bf16Val -> s.slice.devicePtr(stream)
            is SliceInner.F16Val -> s.slice.devicePtr(stream)
            is SliceInner.F32 -> s.slice.devicePtr(stream)
            is SliceInner.I32 -> s.slice.devicePtr(stream)
            is SliceInner.I64 -> s.slice.devicePtr(stream)
        }

    public fun devicePtrMut(stream: CudaStream): Pair<CUdeviceptr, CudaEvent?> =
        when (val s = inner) {
            is SliceInner.Bf16Val -> s.slice.devicePtrMut(stream)
            is SliceInner.F16Val -> s.slice.devicePtrMut(stream)
            is SliceInner.F32 -> s.slice.devicePtrMut(stream)
            is SliceInner.I32 -> s.slice.devicePtrMut(stream)
            is SliceInner.I64 -> s.slice.devicePtrMut(stream)
        }

    override fun copyHostToDevice(src: FloatArray) {
        val s = inner as? SliceInner.F32 ?: throw UgError.Msg("htod dtype mismatch, dst $dtype, src ${DType.F32}")
        device.cudarcStream().memcpyHtod(src.toList(), s.slice).getOrThrow()
    }

    override fun copyHostToDevice(src: IntArray) {
        val s = inner as? SliceInner.I32 ?: throw UgError.Msg("htod dtype mismatch, dst $dtype, src ${DType.I32}")
        device.cudarcStream().memcpyHtod(src.toList(), s.slice).getOrThrow()
    }

    override fun copyHostToDevice(src: LongArray) {
        val s = inner as? SliceInner.I64 ?: throw UgError.Msg("htod dtype mismatch, dst $dtype, src ${DType.I64}")
        device.cudarcStream().memcpyHtod(src.toList(), s.slice).getOrThrow()
    }

    override fun copyHostToDevice(src: Array<F16>) {
        val s = inner as? SliceInner.F16Val ?: throw UgError.Msg("htod dtype mismatch, dst $dtype, src ${DType.F16}")
        device.cudarcStream().memcpyHtod(src.toList(), s.slice).getOrThrow()
    }

    override fun copyHostToDevice(src: Array<BF16>) {
        val s = inner as? SliceInner.Bf16Val ?: throw UgError.Msg("htod dtype mismatch, dst $dtype, src ${DType.BF16}")
        device.cudarcStream().memcpyHtod(src.toList(), s.slice).getOrThrow()
    }

    override fun copyDeviceToHost(dst: FloatArray) {
        val s = inner as? SliceInner.F32 ?: throw UgError.Msg("dtoh dtype mismatch, dst ${DType.F32}, src $dtype")
        val list = device.cudarcStream().cloneDtoh(s.slice).getOrThrow()
        for (i in list.indices) {
            dst[i] = list[i]
        }
    }

    override fun copyDeviceToHost(dst: IntArray) {
        val s = inner as? SliceInner.I32 ?: throw UgError.Msg("dtoh dtype mismatch, dst ${DType.I32}, src $dtype")
        val list = device.cudarcStream().cloneDtoh(s.slice).getOrThrow()
        for (i in list.indices) {
            dst[i] = list[i]
        }
    }

    override fun copyDeviceToHost(dst: LongArray) {
        val s = inner as? SliceInner.I64 ?: throw UgError.Msg("dtoh dtype mismatch, dst ${DType.I64}, src $dtype")
        val list = device.cudarcStream().cloneDtoh(s.slice).getOrThrow()
        for (i in list.indices) {
            dst[i] = list[i]
        }
    }

    override fun copyDeviceToHost(dst: Array<F16>) {
        val s = inner as? SliceInner.F16Val ?: throw UgError.Msg("dtoh dtype mismatch, dst ${DType.F16}, src $dtype")
        val list = device.cudarcStream().cloneDtoh(s.slice).getOrThrow()
        for (i in list.indices) {
            dst[i] = list[i]
        }
    }

    override fun copyDeviceToHost(dst: Array<BF16>) {
        val s = inner as? SliceInner.Bf16Val ?: throw UgError.Msg("dtoh dtype mismatch, dst ${DType.BF16}, src $dtype")
        val list = device.cudarcStream().cloneDtoh(s.slice).getOrThrow()
        for (i in list.indices) {
            dst[i] = list[i]
        }
    }

    override fun toFloatArray(): FloatArray {
        val arr = FloatArray(len)
        copyDeviceToHost(arr)
        return arr
    }

    override fun toIntArray(): IntArray {
        val arr = IntArray(len)
        copyDeviceToHost(arr)
        return arr
    }

    override fun toLongArray(): LongArray {
        val arr = LongArray(len)
        copyDeviceToHost(arr)
        return arr
    }
}

public class Device internal constructor(
    private val context: CudaContext,
    private val stream: CudaStream,
    private val blas: CudaBlas,
) : io.github.kotlinmania.ug.Device {
    public companion object {
        public fun new(deviceIndex: Int): Device {
            val context = CudaContext.new(deviceIndex).getOrThrow()
            val stream = context.defaultStream()
            val blas = CudaBlas.new(stream).getOrThrow()
            return Device(context, stream, blas)
        }

        public fun useGrid(): Boolean = true
    }

    public fun cudarcStream(): CudaStream = stream

    public fun blas(): CudaBlas = blas

    public fun compileCu(cuCode: String, funcName: String): CudaFunction {
        val opts = CompileOptions(useFastMath = true)
        val ptx = compilePtxWithOpts(cuCode, opts).getOrThrow()
        val mdl = context.loadModule(ptx).getOrThrow()
        return mdl.loadFunction(funcName).getOrThrow()
    }

    public fun compilePtx(ptxCode: String, funcName: String): CudaFunction {
        val ptx = Ptx.fromSrc(ptxCode)
        val mdl = context.loadModule(ptx).getOrThrow()
        return mdl.loadFunction(funcName).getOrThrow()
    }

    public fun zeros(len: Int): Slice {
        val slice = stream.allocZeros<Float>(len).getOrThrow()
        return Slice(SliceInner.F32(slice), this)
    }

    public fun sliceFromValues(vs: FloatArray): Slice {
        val slice = allocateUninit(DType.F32, vs.size)
        slice.copyHostToDevice(vs)
        return slice
    }

    public fun sliceFromValues(vs: IntArray): Slice {
        val slice = allocateUninit(DType.I32, vs.size)
        slice.copyHostToDevice(vs)
        return slice
    }

    public fun sliceFromValues(vs: LongArray): Slice {
        val slice = allocateUninit(DType.I64, vs.size)
        slice.copyHostToDevice(vs)
        return slice
    }

    public fun sliceFromValues(vs: Array<F16>): Slice {
        val slice = allocateUninit(DType.F16, vs.size)
        slice.copyHostToDevice(vs)
        return slice
    }

    public fun sliceFromValues(vs: Array<BF16>): Slice {
        val slice = allocateUninit(DType.BF16, vs.size)
        slice.copyHostToDevice(vs)
        return slice
    }

    override fun synchronize() {
        stream.synchronize().getOrThrow()
    }

    override fun allocateUninit(dtype: DType, len: Int): Slice {
        val inner = when (dtype) {
            DType.F32 -> SliceInner.F32(stream.alloc<Float>(len).getOrThrow())
            DType.F16 -> SliceInner.F16Val(stream.alloc<F16>(len).getOrThrow())
            DType.BF16 -> SliceInner.Bf16Val(stream.alloc<BF16>(len).getOrThrow())
            DType.I32 -> SliceInner.I32(stream.alloc<Int>(len).getOrThrow())
            DType.I64 -> SliceInner.I64(stream.alloc<Long>(len).getOrThrow())
        }
        return Slice(inner, this)
    }

    override fun compile(kernel: Kernel, name: String?): Func {
        val cuCode = StringBuilder(8192)
        val kernelId = KernelId.new().asUsize()
        val funcName = if (name != null) "ug_${name}_$kernelId" else "ug_$kernelId"
        gen(cuCode, funcName, kernel)
        val cfg = kernel.launchConfig
        val launchCfg = LaunchConfig(
            gridDim = Dim3(cfg.gridDim.toUInt(), 1u, 1u),
            blockDim = Dim3(cfg.blockDim.toUInt(), 1u, 1u),
            sharedMemBytes = cfg.sharedMem.toUInt(),
        )
        val func = compileCu(cuCode.toString(), funcName)
        return Func(func, stream, launchCfg)
    }

    override fun run(f: Any, args: List<io.github.kotlinmania.ug.Slice>) {
        require(f is Func) { "f must be ugcuda.runtime.Func" }
        if (f.cfg.blockDim.x == 0u || f.cfg.blockDim.y == 0u || f.cfg.blockDim.z == 0u ||
            f.cfg.gridDim.x == 0u || f.cfg.gridDim.y == 0u || f.cfg.gridDim.z == 0u) {
            return
        }
        val builder = f.builder()
        for (arg in args) {
            require(arg is Slice) { "arg must be ugcuda.runtime.Slice" }
            val (ptr, _) = arg.devicePtrMut(f.stream)
            builder.arg(ptr)
        }
        builder.launch(f.cfg).getOrThrow()
    }

    override fun matmul(
        dst: io.github.kotlinmania.ug.Slice,
        lhs: io.github.kotlinmania.ug.Slice,
        rhs: io.github.kotlinmania.ug.Slice,
        b: Int,
        m: Int,
        n: Int,
        k: Int,
        lhsL: Layout,
        rhsL: Layout,
    ) {
        require(dst is Slice && lhs is Slice && rhs is Slice) { "Slices must be ugcuda.runtime.Slice" }
        gemmMatmul(blas, dst, lhs, rhs, listOf(b, m, n, k), lhsL, rhsL)
    }

    override fun matmul(
        dst: io.github.kotlinmania.ug.Slice,
        lhs: io.github.kotlinmania.ug.Slice,
        rhs: io.github.kotlinmania.ug.Slice,
        bmnk: List<Int>,
        lhsL: Layout,
        rhsL: Layout,
    ) {
        require(dst is Slice && lhs is Slice && rhs is Slice) { "Slices must be ugcuda.runtime.Slice" }
        gemmMatmul(blas, dst, lhs, rhs, bmnk, lhsL, rhsL)
    }

    override fun useGrid(): Boolean = true
}
