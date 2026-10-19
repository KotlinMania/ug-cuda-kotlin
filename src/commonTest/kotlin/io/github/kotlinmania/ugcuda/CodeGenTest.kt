package io.github.kotlinmania.ugcuda

import io.github.kotlinmania.ug.Const
import io.github.kotlinmania.ug.DType
import io.github.kotlinmania.ug.lang.Arg
import io.github.kotlinmania.ug.lang.BinaryOp
import io.github.kotlinmania.ug.lang.LaunchConfig
import io.github.kotlinmania.ug.lang.ReduceOp
import io.github.kotlinmania.ug.lang.UnaryOp
import io.github.kotlinmania.ug.lang.ssa.A as SsaA
import io.github.kotlinmania.ug.lang.ssa.Instr
import io.github.kotlinmania.ug.lang.ssa.Kernel
import io.github.kotlinmania.ug.lang.ssa.KernelArg
import io.github.kotlinmania.ug.lang.ssa.Special
import io.github.kotlinmania.ug.lang.ssa.VarId
import io.github.kotlinmania.ugcuda.codegen.fmtF32
import io.github.kotlinmania.ugcuda.codegen.gen
import io.github.kotlinmania.ugcuda.gemm.gemmReducedPrecisionBf16
import io.github.kotlinmania.ugcuda.gemm.gemmReducedPrecisionF16
import io.github.kotlinmania.ugcuda.gemm.gemmReducedPrecisionF32
import io.github.kotlinmania.ugcuda.gemm.setGemmReducedPrecisionBf16
import io.github.kotlinmania.ugcuda.gemm.setGemmReducedPrecisionF16
import io.github.kotlinmania.ugcuda.gemm.setGemmReducedPrecisionF32
import io.github.kotlinmania.ugcuda.runtime.KernelId
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class CodeGenTest {
    @Test
    fun testFmtF32SpecialValues() {
        assertEquals("0. / 0.", buildString { fmtF32(Float.NaN, this) })
        assertEquals("1. / 0.", buildString { fmtF32(Float.POSITIVE_INFINITY, this) })
        assertEquals("-1. / 0.", buildString { fmtF32(Float.NEGATIVE_INFINITY, this) })
        assertEquals("0.0", buildString { fmtF32(0.0f, this) })
        assertEquals("1.5", buildString { fmtF32(1.5f, this) })
    }

    @Test
    fun testReducedPrecisionFlags() {
        assertFalse(gemmReducedPrecisionF32())
        setGemmReducedPrecisionF32(true)
        assertTrue(gemmReducedPrecisionF32())
        setGemmReducedPrecisionF32(false)
        assertFalse(gemmReducedPrecisionF32())

        assertFalse(gemmReducedPrecisionF16())
        setGemmReducedPrecisionF16(true)
        assertTrue(gemmReducedPrecisionF16())
        setGemmReducedPrecisionF16(false)
        assertFalse(gemmReducedPrecisionF16())

        assertFalse(gemmReducedPrecisionBf16())
        setGemmReducedPrecisionBf16(true)
        assertTrue(gemmReducedPrecisionBf16())
        setGemmReducedPrecisionBf16(false)
        assertFalse(gemmReducedPrecisionBf16())
    }

    @Test
    fun testKernelIdMonotonicity() {
        val id1 = KernelId.new().asUsize()
        val id2 = KernelId.new().asUsize()
        assertTrue(id2 > id1)
    }

    @Test
    fun testSimpleAddCodeGen() {
        val kernel = Kernel(
            instrs = listOf(
                Instr.DefineGlobal(0, DType.F32),
                Instr.DefineGlobal(1, DType.F32),
                Instr.DefineGlobal(2, DType.F32),
                Instr.SpecialInstr(Special.ThreadIdx),
                Instr.Load(VarId(0), SsaA.Var(VarId(3)), DType.F32),
                Instr.Load(VarId(1), SsaA.Var(VarId(3)), DType.F32),
                Instr.Binary(BinaryOp.Add, SsaA.Var(VarId(4)), SsaA.Var(VarId(5)), DType.F32),
                Instr.Store(VarId(2), SsaA.Var(VarId(3)), SsaA.Var(VarId(6)), DType.F32),
            ),
            args = listOf(
                KernelArg(Arg.ptr(DType.F32), 0),
                KernelArg(Arg.ptr(DType.F32), 1),
                KernelArg(Arg.ptr(DType.F32), 2),
            ),
            launchConfig = LaunchConfig(gridDim = 1, blockDim = 1024, sharedMem = 0),
        )

        val code = buildString {
            gen(this, "add_kernel", kernel)
        }

        assertTrue(code.contains("extern \"C\" __global__ void add_kernel("))
        assertTrue(code.contains("float* __var0,"))
        assertTrue(code.contains("float* __var1,"))
        assertTrue(code.contains("float* __var2"))
        assertTrue(code.contains("int __var3 = threadIdx.x;"))
        assertTrue(code.contains("float __var4 = __var0[__var3];"))
        assertTrue(code.contains("float __var5 = __var1[__var3];"))
        assertTrue(code.contains("float __var6 = __var4 + __var5;"))
        assertTrue(code.contains("__var2[__var3] = __var6;"))
        assertFalse(code.contains("block_reduce_sum"))
    }

    @Test
    fun testReductionCodeGenIncludesReduceCu() {
        val kernel = Kernel(
            instrs = listOf(
                Instr.DefineGlobal(0, DType.F32),
                Instr.DefineGlobal(1, DType.F32),
                Instr.SpecialInstr(Special.ThreadIdx),
                Instr.Load(VarId(0), SsaA.Var(VarId(2)), DType.F32),
                Instr.ReduceLocal(ReduceOp.Sum, SsaA.Var(VarId(3)), DType.F32),
                Instr.Store(VarId(1), SsaA.Constant(Const.I32(0)), SsaA.Var(VarId(4)), DType.F32),
            ),
            args = listOf(
                KernelArg(Arg.ptr(DType.F32), 0),
                KernelArg(Arg.ptr(DType.F32), 1),
            ),
            launchConfig = LaunchConfig(gridDim = 1, blockDim = 32, sharedMem = 0),
        )

        val code = buildString {
            gen(this, "reduce_kernel", kernel)
        }

        assertTrue(code.contains("block_reduce_sum"))
        assertTrue(code.contains("warp_reduce_sum"))
        assertTrue(code.contains("extern \"C\" __global__ void reduce_kernel("))
        assertTrue(code.contains("float __var4 = block_reduce_sum(__var3);"))
    }
}
