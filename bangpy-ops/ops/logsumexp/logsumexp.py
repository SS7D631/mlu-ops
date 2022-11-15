# Copyright (C) [2021] by Cambricon, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# pylint: disable=missing-docstring, invalid-name, too-many-locals
"""A multi-platform code link example test for BANGPy TCP."""
import bangpy
from bangpy import tcp
from bangpy.script import ty, build_module
import bangpy.eager as eg

DTYPES = [#bangpy.float16,
           bangpy.float32]
TARGET_LIST = ["mlu370-s4"]
KERNEL_NAME = "Logsumexp"


@eg.module
class Logsumexp(object):
    def calc_avaible_nram_count(self):
        return tcp.round_down((self.bp.nram_size - 30 * 1024) // 2, 128)

    def calc_core_process_count(self, data_total_len: ty.int32, task_num: ty.int32, task_id: ty.int32):
        one_core_count = data_total_len // task_num
        remain = data_total_len % task_num
        m_current_core_start = 0
        m_current_core_end = 0
        m_total_count_in_core = 0
        if task_id < remain:
            m_current_core_start = (one_core_count + 1) * task_id
            m_current_core_end = (one_core_count + 1) * (task_id + 1) - 1
            m_total_count_in_core = m_current_core_end - m_current_core_start + 1
        else:
            m_current_core_start = (one_core_count + 1) * \
                remain + one_core_count * (task_id - remain)
            m_current_core_end = (one_core_count + 1) * remain + \
                one_core_count * (task_id - remain) + one_core_count - 1
            m_total_count_in_core = m_current_core_end - m_current_core_start + 1

        self.m_total_count_in_core = m_total_count_in_core
        self.m_current_core_start = m_current_core_start
        self.m_current_core_end = m_current_core_end

    def __init__(self, dtype: ty.string, dtype_size: ty.int32) -> None:
        self.dtype = dtype
        self.dtype_sz = dtype_size 


    def main(self, Gram_tensor: ty.handle,
                    dim_len: ty.int32, h: ty.int32, w: ty.int32,
                    output_len: ty.int32,
                    Gram_border_buf_out: ty.handle, Gram_border_idx_out: ty.handle, Gram_buffer_out: ty.handle
                    ) -> None:
        tgt = tcp.target()
        self.bp = tgt

        gram_tensor = tcp.match_buffer(Gram_tensor, [h * w], dtype=self.dtype)

        border_array_size = 128
        gram_border_buf_out = tcp.match_buffer(Gram_border_buf_out, [border_array_size * 2], dtype=self.dtype)
        gram_border_idx_out = tcp.match_buffer(Gram_border_idx_out, [border_array_size * 2], dtype='int32')
        gram_buffer_out = tcp.match_buffer(Gram_buffer_out, [output_len], dtype=self.dtype)

        for cluster_id in tcp.thread_binding(0, tgt.cluster_num, thread="blockIdx.x"):
            for core_id in tcp.thread_binding(0, tgt.core_num, thread="threadIdx.x"):
                task_num = tgt.cluster_num * tgt.core_num
                task_id = tgt.core_num * cluster_id + core_id
                self.taskId = task_id
                tcp.print("zouni")

                gram_reshape_tensor = gram_tensor[:h * w].reshape([h, w])

                nram_avable_size = self.calc_avaible_nram_count()
                self.nram_process_count = nram_avable_size // self.dtype_sz

                self.nram_calc_buffer = tcp.alloc_buffer(
                    shape=(self.nram_process_count, 1),
                    dtype=self.dtype,
                    scope="nram")

                self.m_buff = tcp.alloc_buffer(
                    shape=(border_array_size * 2,),
                    dtype= tcp.int32, scope="nram"
                )
'''

                self.flat_nram = self.nram_calc_buffer[:self.nram_process_count].reshape([self.nram_process_count, ])

                self.taskId = task_id
                self.calc_core_process_count(h * w, task_num)
                if dim_len > self.nram_process_count:
                    self.calc1(gram_reshape_tensor, gram_border_buf_out,
                               gram_border_idx_out, gram_buffer_out)
                else:
                    if (w * h) // task_num + 1 < dim_len:
                        self.calc1(gram_reshape_tensor, gram_border_buf_out,
                                   gram_border_idx_out, gram_buffer_out)
                    else:
                        self.calc2(gram_reshape_tensor, gram_border_buf_out,
                                   gram_border_idx_out, gram_buffer_out)

                tcp.sync_all()

        def calc2(self, gram_tensor: ty.handle, border_outputs: ty.handle, idx_outputs: ty.handle, outputs: ty.handle):
            return 222

        def calc1(self, gram_tensor: ty.handle, border_outputs: ty.handle, idx_outputs: ty.handle, outputs: ty.handle): 
            once_loop_start = 0
            return 111

'''

@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_add(dtype=None, target=None):
    f = build_module.build(
        Logsumexp(dtype.name, dtype.bytes), target_tag=target, name=KERNEL_NAME
    )
    return f

