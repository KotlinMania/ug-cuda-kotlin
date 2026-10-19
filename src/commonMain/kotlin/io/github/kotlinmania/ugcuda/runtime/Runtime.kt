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
import io.github.kotlinmania.ug.UgException
import io.github.kotlinmania.ug.WithDType
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
            else -> throw UgException("dtype mismatch, expected ${DType.F32}, got ${s.dtype()}")
        }
}

public object ToSliceF16 : ToSlice<F16> {
    override fun slice(s: Slice): CudaSlice<F16> =
        when (val inner = s.inner) {
            is SliceInner.F16Val -> inner.slice
            else -> throw UgException("dtype mismatch, expected ${DType.F16}, got ${s.dtype()}")
        }
}

public object ToSliceBf16 : ToSlice<BF16> {
    override fun slice(s: Slice): CudaSlice<BF16> =
        when (val inner = s.inner) {
            is SliceInner.Bf16Val -> inner.slice
            else -> throw UgException("dtype mismatch, expected ${DType.BF16}, got ${s.dtype()}")
        }
}

public object ToSliceI32 : ToSlice<Int> {
    override fun slice(s: Slice): CudaSlice<Int> =
        when (val inner = s.inner) {
            is SliceInner.I32 -> inner.slice
            else -> throw UgException("dtype mismatch, expected ${DType.I32}, got ${s.dtype()}")
        }
}

public object ToSliceI64 : ToSlice<Long> {
    override fun slice(s: Slice): CudaSlice<Long> =
        when (val inner = s.inner) {
            is SliceInner.I64 -> inner.slice
            else -> throw UgException("dtype mismatch, expected ${DType.I64}, got ${s.dtype()}")
        }
}

public class Slice(
    internal var inner: SliceInner,
    internal val device: Device,
) : io.github.kotlinmania.ug.Slice {
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

    override fun dtype(): DType =
        when (inner) {
            is SliceInner.F32 -> DType.F32
            is SliceInner.F16Val -> DType.F16
            is SliceInner.Bf16Val -> DType.BF16
            is SliceInner.I32 -> DType.I32
            is SliceInner.I64 -> DType.I64
        }

    override fun device(): io.github.kotlinmania.ug.Device = device

    override fun <DT : WithDType> copyHostToDevice(src: List<DT>) {
        when (val s = inner) {
            is SliceInner.Bf16Val -> {
                val list = src.map { it.toBf16() }
                device.cudarcStream().memcpyHtod(list, s.slice).getOrThrow()
            }
            is SliceInner.F16Val -> {
                val list = src.map { it.toF16() }
                device.cudarcStream().memcpyHtod(list, s.slice).getOrThrow()
            }
            is SliceInner.F32 -> {
                val list = src.map { it.toFloat() }
                device.cudarcStream().memcpyHtod(list, s.slice).getOrThrow()
            }
            is SliceInner.I32 -> {
                val list = src.map { it.toInt() }
                device.cudarcStream().memcpyHtod(list, s.slice).getOrThrow()
            }
            is SliceInner.I64 -> {
                val list = src.map { it.toLong() }
                device.cudarcStream().memcpyHtod(list, s.slice).getOrThrow()
            }
        }
    }

    override fun <DT : WithDType> copyDeviceToHost(dst: MutableList<DT>) {
        when (val s = inner) {
            is SliceInner.Bf16Val -> {
                val list = device.cudarcStream().cloneDtoh(s.slice).getOrThrow()
                dst.clear()
                for (item in list) {
                    dst.add(DT.fromBf16(item))
                }
            }
            is SliceInner.F16Val -> {
                val list = device.cudarcStream().cloneDtoh(s.slice).getOrThrow()
                dst.clear()
                for (item in list) {
                    dst.add(DT.fromF16(item))
                }
            }
            is SliceInner.F32 -> {
                val list = device.cudarcStream().cloneDtoh(s.slice).getOrThrow()
                dst.clear()
                for (item in list) {
                    dst.add(DT.fromFloat(item))
                }
            }
            is SliceInner.I32 -> {
                val list = device.cudarcStream().cloneDtoh(s.slice).getOrThrow()
                dst.clear()
                for (item in list) {
                    dst.add(DT.fromInt(item))
                }
            }
            is SliceInner.I64 -> {
                val list = device.cudarcStream().cloneDtoh(s.slice).getOrThrow()
                dst.clear()
                for (item in list) {
                    dst.add(DT.fromLong(item))
                }
            }
        }
    }

    override fun len(): Int =
        when (val s = inner) {
            is SliceInner.F32 -> s.slice.len
            is SliceInner.F16Val -> s.slice.len
            is SliceInner.Bf16Val -> s.slice.len
            is SliceInner.I32 -> s.slice.len
            is SliceInner.I64 -> s.slice.len
        }

    public inline fun <reified DT : WithDType> toVec(): List<DT> {
        val list = mutableListOf<DT>()
        copyDeviceToHost(list)
        return list
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

    public fun <D : WithDType> sliceFromValues(vs: List<D>): Slice {
        val dt = vs.firstOrNull()?.dtype() ?: DType.F32
        val slice = allocateUninit(dt, vs.size)
        slice.copyHostToDevice(vs)
        return slice
    }

    public fun synchronize() {
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

    override fun run(f: Func, args: List<io.github.kotlinmania.ug.Slice>) {
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
        bmnk: List<Int>,
        lhsL: Layout,
        rhsL: Layout,
    ) {
        require(dst is Slice && lhs is Slice && rhs is Slice) { "Slices must be ugcuda.runtime.Slice" }
        gemmMatmul(blas, dst, lhs, rhs, bmnk, lhsL, rhsL)
    }

    override fun useGrid(): Boolean = true
}
