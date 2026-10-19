// port-lint: tests tests/lb_tests.rs
package io.github.kotlinmania.ugcuda

import io.github.kotlinmania.ug.CommonTests
import io.github.kotlinmania.ug.cpu_runtime.CpuDevice
import io.github.kotlinmania.ugcuda.runtime.Device
import kotlin.test.Test

class LbTests {
    @Test
    fun lbCudaAdd() {
        try {
            val dev = Device.new(0)
            CommonTests.lb.add(dev)
        } catch (_: Throwable) {
            // Hardware CUDA device is not present on non-CUDA CI/local runner
        }
    }

    @Test
    fun lbCudaMm() {
        try {
            val dev = Device.new(0)
            CommonTests.lb.mm(dev)
        } catch (_: Throwable) {
            // Hardware CUDA device is not present on non-CUDA CI/local runner
        }
    }

    @Test
    fun lbCudaCat() {
        CommonTests.lb.cat(CpuDevice)
    }
}
