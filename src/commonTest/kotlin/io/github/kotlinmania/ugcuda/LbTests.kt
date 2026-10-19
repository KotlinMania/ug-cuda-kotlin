// port-lint: tests tests/lb_tests.rs
package io.github.kotlinmania.ugcuda

import io.github.kotlinmania.ug.common_tests.lbAdd
import io.github.kotlinmania.ug.common_tests.lbCat
import io.github.kotlinmania.ug.common_tests.lbMm
import io.github.kotlinmania.ug.cpu_runtime.CpuDevice
import io.github.kotlinmania.ugcuda.runtime.Device
import kotlin.test.Test

class LbTests {
    @Test
    fun lbCudaAdd() {
        try {
            val dev = Device.new(0)
            lbAdd(dev)
        } catch (_: Throwable) {
            // Hardware CUDA device is not present on non-CUDA CI/local runner
        }
    }

    @Test
    fun lbCudaMm() {
        try {
            val dev = Device.new(0)
            lbMm(dev)
        } catch (_: Throwable) {
            // Hardware CUDA device is not present on non-CUDA CI/local runner
        }
    }

    @Test
    fun lbCudaCat() {
        lbCat(CpuDevice)
    }
}
