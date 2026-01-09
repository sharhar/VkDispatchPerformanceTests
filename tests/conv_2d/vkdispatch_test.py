import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import run_test, Config

import vkdispatch as vd
import vkdispatch.codegen as vc

def io_count(elem_count: int, fft_size: int) -> int:
    return 10.0 + (fft_size * fft_size) / elem_count

def make_convolution_function(buffer_shape, kernel_transposed: bool):
    with vd.fft.fft_context(buffer_shape, axis=1) as ctx:
        args = ctx.declare_shader_args([vc.Buffer[vc.c64], vc.Buffer[vc.c64]])

        for read_op in vd.fft.global_reads_iterator(ctx.registers):
            read_op.read_from_buffer(args[0])

        ctx.execute(inverse=False)
        ctx.register_shuffle()

        kernel_registers = ctx.allocate_registers("kernel")

        for read_op in vd.fft.global_reads_iterator(kernel_registers, format_transposed=kernel_transposed):
            read_op.read_from_buffer(args[1])

        for reg_idx in range(ctx.registers.count):
            ctx.registers[reg_idx][:] = vc.mult_complex(ctx.registers[reg_idx], kernel_registers[reg_idx].conjugate())

        ctx.execute(inverse=True)

        for write_op in vd.fft.global_writes_iterator(ctx.registers):
            write_op.write_to_buffer(args[0])
    
    return ctx.get_callable()

def test_function(config: Config,
                    fft_size: int,
                    buffer: vd.Buffer,
                    kernel: vd.Buffer):
    assert kernel.shape == buffer.shape

    vd.fft.fft(buffer)
    make_convolution_function(buffer.shape, kernel_transposed=False)(buffer, kernel)
    vd.fft.ifft(buffer)

def test_function_transpose(config: Config,
                    fft_size: int,
                    buffer: vd.Buffer,
                    kernel: vd.Buffer):
    assert kernel.size >= vd.fft.get_transposed_size(buffer.shape, axis=1)
    vd.fft.fft(buffer)
    make_convolution_function(buffer.shape, kernel_transposed=True)(buffer, kernel)
    vd.fft.ifft(buffer)

@vd.shader("buff.size")
def convolve_naive(buff: vc.Buff[vc.c64], kernel: vc.Buff[vc.c64], fft_size: vc.Const[vc.u32]):
    ind = vc.global_invocation_id().x
    buff[ind] = vc.mult_complex(buff[ind], kernel[ind % (fft_size * fft_size)].conjugate())

def test_function_naive(config: Config,
                    fft_size: int,
                    buffer: vd.Buffer,
                    kernel: vd.Buffer):
    vd.fft.fft2(buffer)
    convolve_naive(buffer, kernel, fft_size)
    vd.fft.ifft2(buffer)

if __name__ == "__main__":
    run_test("vkdispatch", 11, test_function)
    run_test("vkdispatch_transpose", 11, test_function_transpose)
    run_test("vkdispatch_naive", 11, test_function_naive)