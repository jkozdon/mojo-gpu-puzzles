from memory import UnsafePointer

# ANCHOR: softmax_gpu_kernel
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from math import exp
from utils.numerics import max_finite, min_finite


alias SIZE = 128
alias TPB = 128
alias BLOCKS_PER_GRID = (1, 1)
alias THREADS_PER_BLOCK = (TPB, 1)
alias layout = Layout.row_major(SIZE)
alias dtype = DType.float32

fn log2_pow2(n: Int) -> Int:
    var result = 0
    var value = n
    while value > 1:
        value = value >> 1
        result += 1
    return result

fn softmax_gpu_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[mut=True, dtype, layout],
    input: LayoutTensor[mut=False, dtype, layout],
):
    g_x = block_dim.x * block_idx.x + thread_idx.x
    l_x = thread_idx.x
    s_tmp = tb[dtype]().row_major[TPB]().shared().alloc()
    x = input[g_x] if g_x < input_size else input.element_type(0).MIN
    s_tmp[l_x] = x
    barrier()


    stride = TPB // 2
    l_tmp = x
    while stride > 0:
        if l_x < stride:
            l_tmp = max(l_tmp, s_tmp[l_x + stride])
            s_tmp[l_x] = l_tmp
        stride = stride // 2
        barrier()
    x_max = s_tmp[0]

    l_val = exp(x - x_max)
    s_tmp[l_x] = l_val
    l_tmp = l_val
    stride = TPB // 2
    while stride > 0:
        if l_x < stride:
            l_tmp = l_tmp + s_tmp[l_x + stride]
            s_tmp[l_x] = l_tmp
        stride = stride // 2
        barrier()
    l_val /= s_tmp[0]
    if g_x < input_size:
        out[g_x] = l_val


# ANCHOR_END: softmax_gpu_kernel


# ANCHOR: softmax_cpu_kernel
fn softmax_cpu_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, layout, MutableAnyOrigin],
    input: LayoutTensor[dtype, layout, MutableAnyOrigin],
):
    x_max = out.element_type(0).MIN
    for i in range(input_size):
        x_max = max(x_max, input[i])

    x_sum = out.element_type(0)
    for i in range(input_size):
        res = exp(input[i] - x_max)
        x_sum += res
        out[i] = res

    for i in range(input_size):
        out[i] /= x_sum


# ANCHOR_END: softmax_cpu_kernel

import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor


@compiler.register("softmax")
struct SoftmaxCustomOp:
    @staticmethod
    fn execute[
        target: StaticString,  # "cpu" or "gpu"
        input_size: Int,
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[dtype=dtype, rank=1],
        input: InputTensor[dtype = output.dtype, rank = output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        # Note: rebind is necessary now but it shouldn't be!
        var output_tensor = rebind[
            LayoutTensor[dtype, layout, MutableAnyOrigin]
        ](output.to_layout_tensor())
        var input_tensor = rebind[
            LayoutTensor[dtype, layout, MutableAnyOrigin]
        ](input.to_layout_tensor())
        alias layout = input_tensor.layout

        @parameter
        if target == "gpu":
            gpu_ctx = ctx.get_device_context()
            # making sure the output tensor is zeroed out before the kernel is called
            gpu_ctx.enqueue_memset(
                DeviceBuffer[output.dtype](
                    gpu_ctx,
                    rebind[UnsafePointer[Scalar[output.dtype]]](
                        output_tensor.ptr
                    ),
                    input_size,
                    owning=False,
                ),
                0,
            )

            gpu_ctx.enqueue_function[
                softmax_gpu_kernel[layout, input_size, dtype]
            ](
                output_tensor,
                input_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=(TPB, 1),
            )

        elif target == "cpu":
            softmax_cpu_kernel[layout, input_size, dtype](
                output_tensor, input_tensor
            )
        else:
            raise Error("Unsupported target: " + target)
