// port-lint: source src/code_gen.rs
package io.github.kotlinmania.ugcuda.codegen

import io.github.kotlinmania.ug.Const
import io.github.kotlinmania.ug.DType
import io.github.kotlinmania.ug.UgException
import io.github.kotlinmania.ug.lang.ssa.A as SsaA
import io.github.kotlinmania.ug.lang.ssa.BinaryOp
import io.github.kotlinmania.ug.lang.ssa.Instr
import io.github.kotlinmania.ug.lang.ssa.Kernel
import io.github.kotlinmania.ug.lang.ssa.ReduceOp
import io.github.kotlinmania.ug.lang.ssa.Special
import io.github.kotlinmania.ug.lang.ssa.Type
import io.github.kotlinmania.ug.lang.ssa.UnaryOp
import io.github.kotlinmania.ug.lang.ssa.VarId

public class V(public val id: VarId) {
    public fun fmt(f: Appendable) {
        f.append("__var").append(id.asUsize().toString())
    }

    override fun toString(): String = "__var${id.asUsize()}"
}

public class C(public val constVal: Const) {
    public fun fmt(f: Appendable) {
        when (constVal) {
            is Const.BF16 -> fmtF32(constVal.toFloat(), f)
            is Const.F16 -> fmtF32(constVal.toFloat(), f)
            is Const.F32 -> fmtF32(constVal.value, f)
            is Const.I32 -> f.append(constVal.value.toString())
            is Const.I64 -> f.append(constVal.value.toString())
        }
    }

    override fun toString(): String = buildString { fmt(this) }
}

public fun fmtF32(v: Float, f: Appendable) {
    when {
        v.isNaN() -> f.append("0. / 0.")
        v.isInfinite() && v > 0.0f -> f.append("1. / 0.")
        v.isInfinite() -> f.append("-1. / 0.")
        else -> f.append(v.toString())
    }
}

public class A(public val aVal: SsaA) {
    public fun fmt(f: Appendable) {
        when (aVal) {
            is SsaA.Var -> V(aVal.id).fmt(f)
            is SsaA.Constant -> C(aVal.value).fmt(f)
        }
    }

    override fun toString(): String = buildString { fmt(this) }
}

public class D(public val dtype: DType) {
    public fun fmt(f: Appendable) {
        val str = when (dtype) {
            DType.BF16 -> "__nv_bfloat16"
            DType.F16 -> "__half"
            DType.F32 -> "float"
            DType.I32 -> "int"
            DType.I64 -> "long long"
        }
        f.append(str)
    }

    override fun toString(): String = buildString { fmt(this) }
}

public fun gen(w: Appendable, funcName: String, kernel: Kernel) {
    val instrs = kernel.instrs
    val containsReduceLocal = instrs.any { it is Instr.ReduceLocal }
    if (containsReduceLocal) {
        w.append(REDUCE_CU)
    }
    w.append("extern \"C\" __global__ void ").append(funcName).append("(\n")
    for ((argIdx, argPair) in kernel.args.withIndex()) {
        val (arg, varId) = argPair
        val isLast = argIdx == kernel.args.size - 1
        val delim = if (isLast) "" else ","
        val tyStr = when (val t = arg.type) {
            is Type.Value -> D(t.dtype).toString()
            is Type.Ptr -> "${D(t.dtype)}*"
        }
        w.append("  ").append(tyStr).append(" ").append(V(VarId(varId)).toString()).append(delim).append("\n")
    }
    w.append(") {\n")

    var depth = 0
    for ((varIdIdx, instr) in instrs.withIndex()) {
        val varId = V(VarId(varIdIdx))
        val ind = indent(depth)
        when (instr) {
            is Instr.DefineGlobal -> {}
            is Instr.DefineLocal -> {
                w.append(ind).append("__shared__ ").append(D(instr.dtype).toString()).append(" ").append(varId.toString()).append("[").append(instr.size.toString()).append("];\n")
            }
            is Instr.DefineAcc -> {
                w.append(ind).append(D(instr.value.dtype).toString()).append(" ").append(varId.toString()).append(" = ").append(C(instr.value).toString()).append(";\n")
            }
            is Instr.Constant -> {
                w.append(ind).append(D(instr.value.dtype).toString()).append(" ").append(varId.toString()).append(" = ").append(C(instr.value).toString()).append(";\n")
            }
            is Instr.If -> {
                w.append(ind).append("if (").append(A(instr.cond).toString()).append(") {\n")
                depth += 1
            }
            is Instr.Range -> {
                w.append(ind).append("for (int ").append(varId.toString()).append(" = ").append(A(instr.lo).toString()).append("; ").append(varId.toString()).append(" < ").append(A(instr.up).toString()).append("; ").append(varId.toString()).append("+=").append(instr.step.toString()).append(") {\n")
                depth += 1
            }
            is Instr.EndIf, is Instr.EndRange -> {
                if (depth == 0) {
                    throw UgException("unmatched EndRange")
                }
                depth -= 1
                val closeIndent = indent(depth - 1)
                w.append(closeIndent).append("}\n")
            }
            is Instr.Load -> {
                w.append(ind).append(D(instr.dtype).toString()).append(" ").append(varId.toString()).append(" = ").append(V(instr.src).toString()).append("[").append(A(instr.offset).toString()).append("];\n")
            }
            is Instr.Assign -> {
                w.append(ind).append(V(instr.dst).toString()).append(" = ").append(A(instr.src).toString()).append(";\n")
            }
            is Instr.Store -> {
                w.append(ind).append(V(instr.dst).toString()).append("[").append(A(instr.offset).toString()).append("] = ").append(A(instr.value).toString()).append(";\n")
            }
            is Instr.Binary -> {
                val opStr = when (instr.op) {
                    BinaryOp.Add -> "${A(instr.lhs)} + ${A(instr.rhs)}"
                    BinaryOp.Mul -> "${A(instr.lhs)} * ${A(instr.rhs)}"
                    BinaryOp.Sub -> "${A(instr.lhs)} - ${A(instr.rhs)}"
                    BinaryOp.Div -> "${A(instr.lhs)} / ${A(instr.rhs)}"
                    BinaryOp.Min -> "min(${A(instr.lhs)}, ${A(instr.rhs)})"
                    BinaryOp.Max -> "max(${A(instr.lhs)}, ${A(instr.rhs)})"
                    BinaryOp.Mod -> "${A(instr.lhs)} % ${A(instr.rhs)}"
                }
                w.append(ind).append(D(instr.dtype).toString()).append(" ").append(varId.toString()).append(" = ").append(opStr).append(";\n")
            }
            is Instr.Unary -> {
                val opStr = when (val op = instr.op) {
                    UnaryOp.Exp -> "__expf"
                    UnaryOp.Sqrt -> "sqrtf"
                    UnaryOp.Sin -> "sinf"
                    UnaryOp.Cos -> "cosf"
                    UnaryOp.Neg -> "-"
                    UnaryOp.Id -> ""
                    is UnaryOp.Cast -> when (instr.dtype) {
                        DType.BF16 -> "static_cast<__nv_bfloat16>"
                        DType.F16 -> "static_cast<__half>"
                        DType.F32 -> "static_cast<float>"
                        DType.I32 -> "static_cast<int>"
                        DType.I64 -> "static_cast<long long>"
                    }
                }
                w.append(ind).append(D(instr.dtype).toString()).append(" ").append(varId.toString()).append(" = ").append(opStr).append("(").append(A(instr.arg).toString()).append(");\n")
            }
            is Instr.SpecialInstr -> {
                when (instr.special) {
                    Special.ThreadIdx -> w.append(ind).append("int ").append(varId.toString()).append(" = threadIdx.x;\n")
                    Special.BlockIdx -> w.append(ind).append("int ").append(varId.toString()).append(" = blockIdx.x;\n")
                }
            }
            is Instr.Barrier -> {
                w.append(ind).append("__syncthreads();\n")
            }
            is Instr.ReduceLocal -> {
                val opStr = when (instr.op) {
                    ReduceOp.Sum -> "block_reduce_sum"
                    ReduceOp.Min -> "block_reduce_min"
                    ReduceOp.Max -> "block_reduce_max"
                }
                w.append(ind).append(D(instr.dtype).toString()).append(" ").append(varId.toString()).append(" = ").append(opStr).append("(").append(A(instr.arg).toString()).append(");\n")
            }
        }
    }
    w.append("}\n")
    if (depth > 0) {
        throw UgException("unmatched Range")
    }
}
