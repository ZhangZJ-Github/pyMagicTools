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
import time
from typing import *

import numpy

import _base
from _logging import logger


def str_list_to_file(sl: List[str], filename: str):
    t = time.time()
    s = ''.join(sl)
    with open(filename, "w") as f:
        f.write(s)
    logger.info("dt = %.2f" % (time.time() - t))


class FLD_tool:
    def __init__(self, filename):
        t = time.time()
        self.filename = filename
        self.linelist = []
        with open(self.filename, 'r') as f:
            self.linelist = f.readlines()
        # logger.info("dt = %.2f" % (time.time() - t))
        # t = time.time()
        self.parse_important_indexes()
        self.x1 = self._get_data(self.indexes_of_x1s[0], self.n_line_x1_each_t)
        self.x2 = self._get_data(self.indexes_of_x2s[0], self.n_line_x2_each_t)

        assert len(self.x1) == self.n_x1_each_t

        logger.info("dt = %.2f" % (time.time() - t))

    def _get_data(self, index_of_1st_line, n_lines):
        return numpy.fromstring(''.join(self.linelist[index_of_1st_line:index_of_1st_line + n_lines]), dtype=float,
                                sep=' ')

    def get_values(self, i):
        t = self.get_time(i)
        values = self._get_data(self.indexes_of_values[i], self.n_line_values_each_t)
        return t, values, self.x1, self.x2

    def get_time(self, i):
        return ''.join(re.findall(r' SOLIDFILL\s+' + _base.FrequentUsedPatter.float,
                                  self.linelist[self.indexes_of_titles[i] - 11])[0])

    def parse_important_indexes(self):
        """
        根据文件内容的前面几行，解析出我们关心的一些重要信息所在的行数
        :return:
        """
        n_file_info = 1  # 用于描述整个文件的编码方式的行数
        n_info_each_t = 20 - 2
        i_value_range_each_t = 19 - 2  # 每个时间片中，用于设置场显示范围的那一行的相对位置
        i_time_each_t = 14 - 2
        info = self.linelist[7]
        strs_n_x1_n_x2 = re.findall(r"[0-9]+_BY_", info)
        self.n_x1_each_t, self.n_x2_each_t = (int(strs_n_x1_n_x2[i][:-4]) for i in range(2))
        self.n_values_each_t = self.n_x1_each_t * self.n_x2_each_t
        n_values_each_line = 5
        self.n_line_x1_each_t, self.n_line_x2_each_t, self.n_line_values_each_t = numpy.ceil(
            numpy.array((self.n_x1_each_t, self.n_x2_each_t, self.n_values_each_t)) / numpy.array(
                (n_values_each_line,) * 3)).astype(
            int)
        self.n_lines_each_time = n_info_each_t + 2 + self.n_line_x1_each_t + self.n_line_x2_each_t + self.n_line_values_each_t
        self.n_blocks = (len(self.linelist) - 1) / self.n_lines_each_time
        arr = self.n_lines_each_time * numpy.arange(self.n_blocks).astype(int) + n_file_info
        self.indexes_of_titles = arr + i_time_each_t  # 涉及时间片信息的（绝对）行号
        self.indexes_of_value_range = arr + i_value_range_each_t
        self.indexes_of_x1s = self.indexes_of_value_range + 2
        self.indexes_of_x2s = self.indexes_of_x1s + self.n_line_x1_each_t
        self.indexes_of_values = self.indexes_of_x2s + 1 + self.n_line_x2_each_t

        # return indexes_of_titles, indexes_of_value_range, n_x1_each_t, n_x2_each_t, n_blocks, indexes_of_x1s, indexes_of_x2s, indexes_of_values

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
    filename = r"F:\MagicFiles\CherenkovAcc\cascade\Coax-2-cascade-higher-gradient-06.fld"
    fld = FLD_tool(filename)
    fld.replace_value_range(-1e6, 1e6, )
