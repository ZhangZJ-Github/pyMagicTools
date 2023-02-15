# -*- coding: utf-8 -*-
# @Time    : 2023/2/15 15:10
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : set_unity_range.py
# @Software: PyCharm
"""
脚本功能：
为MAGIC导出的场文件自动设置范围，以便显示
"""
import re
from typing import *
from _logging import logger
import time
import numpy


def str_list_to_file(sl: List[str], filename: str):
    t= time.time()
    s = ''.join(sl)
    with open(filename, "w") as f:
        f.write(s)
    logger.info("dt = %.2f"%(time.time()-t))


class FLD_tool:
    def __init__(self, filename):
        t = time.time()
        self.filename = filename
        self.linelist = ""
        with open(self.filename, 'r') as f:
            self.linelist = f.readlines()
        logger.info("dt = %.2f" % (time.time() - t))
        t = time.time()
        self.indexes_of_timestep, self.indexes_of_value_range, self.n_x1_each_t, self.n_x2_each_t, self.n_timesteps = self.important_indexes(
            self.linelist)
        logger.info("dt = %.2f" % (time.time() - t))

    @staticmethod
    def important_indexes(line_list: list):
        """
        根据文件内容的前面几行，解析出我们关心的一些重要信息所在的行数
        :return:
        """
        n_file_info = 1  # 用于描述整个文件的编码方式的行数
        n_info_each_t = 20 - 2
        i_value_range_each_t = 19 - 2  # 每个时间片中，用于设置场显示范围的那一行的相对位置
        i_time_each_t = 14 - 2
        info = line_list[7]
        strs_n_x1_n_x2 = re.findall(r"[0-9]+_BY_", info)
        n_x1_each_t, n_x2_each_t = (int(strs_n_x1_n_x2[i][:-4]) for i in range(2))
        n_values_each_t = n_x1_each_t * n_x2_each_t
        n_values_each_line = 5
        n_line_x1_each_t, n_line_x2_each_t, n_line_values_each_t = numpy.ceil(
            numpy.array((n_x1_each_t, n_x2_each_t, n_values_each_t)) / numpy.array((n_values_each_line,) * 3)).astype(
            int)
        n_lines_each_time = n_info_each_t + 2 + n_line_x1_each_t + n_line_x2_each_t + n_line_values_each_t
        n_timesteps = (len(line_list) - 1) / n_lines_each_time
        arr = n_lines_each_time * numpy.arange(n_timesteps).astype(int) + n_file_info
        indexes_of_timestep = arr + i_time_each_t  # 涉及时间片信息的（绝对）行号
        indexes_of_value_range = arr + i_value_range_each_t
        return indexes_of_timestep, indexes_of_value_range, n_x1_each_t, n_x2_each_t, n_timesteps

    def replace_value_range(self, new_value_range_start, new_value_range_end, new_value_range_step=None):

        n_steps_max = 10

        if new_value_range_step is None or (
                new_value_range_end - new_value_range_start) / new_value_range_step > n_steps_max:
            new_value_range_step = (new_value_range_end - new_value_range_start) / n_steps_max
        t = time.time()
        for i in self.indexes_of_value_range:
            self.linelist[i] = "  $$$RANGE(%.4e, %.4e, %.4e)\n" % (
            new_value_range_start, new_value_range_end, new_value_range_step)
        str_list_to_file(self.linelist, self.filename)
        logger.info("dt = %.2f" % (time.time() - t))



if __name__ == '__main__':
    filename = r"F:\MagicFiles\CherenkovAcc\Coax-inner_cone_and_sheet.fld"
    fld = FLD_tool(filename)
    fld.replace_value_range(-1e6, 1e6, )
