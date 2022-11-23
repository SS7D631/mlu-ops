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

        self.m_norm_value = 0.0
        self._norm_oper_count = 0
        self.m_lc_value = 0.0
        self._lc_oper_count = 0

        self.m_size = 0
        self.m_buff = None

    def nm_reset(self):
        self.m_norm_value = 0.0
        self._norm_oper_count = 0

    def lc_reset(self):
        self.m_lc_value = 0.0
        self._lc_oper_count = 0

    def nm_add_buffer(self, buffer: ty.handle, start_index: ty.int32, end_index: ty.int32):
        if self._norm_oper_count == 0:
            self.m_norm_value = self.calc_buffer(buffer, start_index, end_index)
        else:
            ret_value = self.calc_buffer(buffer, start_index, end_index)
            tmp_calc_value = self.m_norm_value
            tmp_ret = self.calc_value(tmp_calc_value, ret_value)
            self.m_norm_value = tmp_ret

        self._norm_oper_count += 1
        return self.m_value


    def lc_add_buffer(self, buffer: ty.handle, start_index, end_index):
        if self._lc_oper_count == 0:
            self.m_lc_value = self.calc_buffer(buffer, start_index, end_index)
        else:
            ret_value = self.calc_buffer(buffer, start_index, end_index)
            tmp_calc_value = self.m_lc_value
            tmp_ret = self.calc_value(tmp_calc_value, ret_value)
            self.m_lc_value = tmp_ret

        self._lc_oper_count += 1
        return self.m_lc_value




    def calc_buffer(self, buffer: ty.handle, start_index: ty.int32, end_index: ty.int32):
        natural_base = 2.7182818284590452353602874713526624977572470936999
        const_one = 1
        max_threshold_valu = 88.722008965395851698332450562653
        min_threshold_valu = -87.332719095296162600686375692197
        data_length = end_index - start_index
        sub_value = 0.0
        sum_value = buffer[start_index]
        #with self.bp.for_range(0, data_length - 1) as i:
        for i in range(data_length - 1):
            sub_value = sum_value - buffer[i + 1]
            if sub_value <= max_threshold_valu and sub_value >= min_threshold_valu:
                sum_value = tcp.scalar_pow(natural_base, sub_value) + const_one
                sum_value = tcp.scalar_log(sum_value) / self.bp.scalar_log(natural_base)
                sum_value = sum_value + buffer[i + 1]
            else:
                if sub_value < min_threshold_valu:
                    sum_value = buffer[i + 1]
        return sum_value










    def add(self, index: ty.int32):
        self.m_buff[self.m_size] = index
        self.m_size += 1

    def check_in(self, index: ty.int32):
        for i in range(self.m_size):
            if index == self.m_buff[i]:
                return 1
        return 0

    def get_calc_loop_count(self):
        return (self.m_total_count_in_core + self.nram_process_count - 1) // self.nram_process_count

    def copy_from_2d_tensor(self, dst: ty.handle, offset_dst: ty.int32, src: ty.handle, offset_src: ty.int32, dim_len: ty.int32, width: ty.int32, cp_len: ty.int32):
        big_row = offset_src // (width * dim_len)

        m = offset_src % dim_len + big_row * dim_len

        big_n = offset_src // dim_len
        n = big_n % width

        if offset_dst != offset_dst + cp_len // 2:
            tcp.memcpy(dst[offset_dst:offset_dst + cp_len // 2, 0:1],
                           src[m:m + cp_len // 2, n:n + 1])

        if offset_dst + cp_len // 2 != offset_dst + cp_len:
            tcp.memcpy(dst[offset_dst + cp_len // 2:offset_dst + cp_len, 0:1],
                           src[m + cp_len // 2:m + cp_len, n:n + 1])

    def calc1(self, gram_tensor: ty.handle, border_outputs: ty.handle, idx_outputs: ty.handle, outputs: ty.handle):
        once_loop_start = 0
        norm_offset = self.m_current_core_start % self.dim_len
        
        #norm_value = LogSumCalcer(self.bp, self.dtype)

        self.calc_size = self.nram_process_count
        once_norm_ok = 0
        cp_data_len = 0
        loop_count = self.get_calc_loop_count()
        for i in range(loop_count):
            once_loop_start = self.m_current_core_start + self.nram_process_count * i
            if i == loop_count - 1:
                self.calc_size = self.m_total_count_in_core % self.nram_process_count
                if self.calc_size == 0:
                    self.calc_size = self.nram_process_count

            norm_offset = once_loop_start % self.dim_len
            expect_cp_len = self.dim_len - norm_offset

            if expect_cp_len > self.calc_size:
                expect_cp_len = self.calc_size
                self.copy_from_2d_tensor(self.nram_calc_buffer, 0, gram_tensor, once_loop_start, self.dim_len, self.w, expect_cp_len)
                cp_data_len = cp_data_len + expect_cp_len
                
                #norm_value.add_buffer(self.flat_nram, 0, expect_cp_len)
                
                '''
                if i == loop_count - 1:
                    index = get_norm_index(once_loop_start + expect_cp_len, self.para.dim_len)
                    with self.bp.if_scope(once_norm_ok == 0):
                        border_outputs[self.bp.taskId * 2] = \
                            norm_value.m_value
                        idx_outputs[self.bp.taskId * 2] = index
                    with self.bp.else_scope():
                        border_outputs[self.bp.taskId * 2 + 1] = norm_value.m_value
                        idx_outputs[self.bp.taskId * 2 + 1] = index
                        '''




    def calc2(self, gram_tensor: ty.handle, border_outputs: ty.handle, idx_outputs: ty.handle, outputs: ty.handle):
        return 12

    def calc_value(self, x, y):
        natural_base = 2.718281
        max_threshold_valu = 88.72200
        min_threshold_valu = -87.33271
        const_one = 1
        scalar_res = y - x
        if scalar_res <= max_threshold_valu and scalar_res >= min_threshold_valu:
            scalar_res = tcp.scalar_pow(natural_base, scalar_res)
            scalar_res = scalar_res + const_one
            scalar_res = tcp.scalar_log(scalar_res) / tcp.scalar_log(natural_base)
            scalar_res = scalar_res + x
        else:
            if scalar_res > max_threshold_valu:
                scalar_res = y
            else:
                scalar_res = x

        return scalar_res


    def main(self, Gram_tensor: ty.handle,
                    dim_len: ty.int32, h: ty.int32, w: ty.int32,
                    output_len: ty.int32,
                    Gram_border_buf_out: ty.handle, Gram_border_idx_out: ty.handle, Gram_buffer_out: ty.handle
                    ) -> None:
        tgt = tcp.target()
        self.bp = tgt
        self.w = w

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

                gram_reshape_tensor = gram_tensor[:h * w].reshape([h, w])

                nram_avable_size = self.calc_avaible_nram_count()
                self.nram_process_count = nram_avable_size // self.dtype_sz

                self.nram_calc_buffer = tcp.alloc_buffer(
                    shape=(self.nram_process_count, 1),
                    dtype=self.dtype,
                    scope="nram")

                self.m_buff = tcp.alloc_buffer(
                    shape=(border_array_size * 2,),
                    dtype= 'int32', scope="nram"
                )

                self.flat_nram = self.nram_calc_buffer[:self.nram_process_count].reshape([self.nram_process_count, ])

                self.dim_len = dim_len
                self.taskId = task_id
                self.calc_core_process_count(h * w, task_num, task_id)
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

                self.m_buff = tcp.alloc_buffer(
                        shape = (border_array_size * 2,),
                        dtype = 'int32', scope="nram"
                    )

                ret2 = 0.0
                if task_id == 0:
                    for i in range(border_array_size):
                        index1 = gram_border_idx_out[2 * i]
                        index2 = gram_border_idx_out[2 * i + 1]
                        norm_value1 = gram_border_buf_out[2 * i]
                        norm_value2 = gram_border_buf_out[2 * i + 1]

                        if index1 >= 0:
                            chk_ret = self.check_in(index1)
                            if chk_ret == 0:
                                gram_buffer_out[index1] = norm_value1
                                self.add(index1)
                            else:
                                ret2 = self.calc_value(gram_buffer_out[index1], norm_value1)
                                gram_buffer_out[index1] = ret2

                        if index2 >= 0:
                            chk_ret = self.check_in(index2)
                            if chk_ret == 0:
                                gram_buffer_out[index2] = norm_value2
                                self.add(index2)
                            else:
                                ret2 = self.calc_value(gram_buffer_out[index2], norm_value2)
                                gram_buffer_out[index2] = ret2
                                    





@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_add(dtype=None, target=None):
    f = build_module.build(
        Logsumexp(dtype.name, dtype.bytes), target_tag=target, name=KERNEL_NAME
    )
    return f

