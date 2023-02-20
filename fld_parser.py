# -*- coding: utf-8 -*-
# @Time    : 2023/2/19 21:19
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : fld_parser.py
# @Software: PyCharm
import re
import time

import numpy

import _base
from _logging import logger


class FLD(_base.ParserBase):

    def __init__(self, filename):
        t = time.time()

        super(FLD, self).__init__(filename)
        self.x1x2grid = self.build_x1_x2_meshgrid(0)[:2]
        self.field_values_all_t = None
        logger.info("dt = %.2f" % (time.time() - t))

    def get_time(self, i):
        """
        Get time of the i-th block
        获取第i个block的时间
        :param i:
        :return:
        """
        return float(''.join(re.findall(r' SOLIDFILL\s+' + _base.FrequentUsedPatter.float, self.block_list[i])[0]))

    def _block_splitter(self, i):
        """
        :param i:
        :return:
        """
        linelist = self.block_list[i].splitlines(True)
        t = self.get_time(i)
        n_x1, n_x2 = (int(s) for s in re.findall(r'ARRAY_\s+([0-9]+)_BY_\s+([0-9]+)_BY_', linelist[8 - 2])[0])
        n_values = n_x1 * n_x2
        n_values_each_line = 5
        n_line_x1, n_line_x2, n_line_values = numpy.ceil(
            numpy.array((n_x1, n_x2, n_values)) / numpy.array(
                (n_values_each_line,) * 3)).astype(
            int)
        title = linelist[4 - 2][:-1]  # 不含时间信息的标题
        i_x1_start = 21 - 2
        i_x2_start = i_x1_start + n_line_x1 + 1
        i_field_value_start = i_x2_start + n_line_x2

        return linelist, t, title, n_line_x1, n_line_x2, n_line_values, i_x1_start, i_x2_start, i_field_value_start, n_x1, n_x2

    def get_values_by_index(self, i):
        linelist, t, title, n_line_x1, n_line_x2, n_line_values, i_x1_start, i_x2_start, i_field_value_start, n_x1, n_x2 = self._block_splitter(
            i)
        field_values = self._get_data(linelist, i_field_value_start, n_line_values)
        field_ranges = [
            float(s) for s in re.findall(
            r'RANGE\(' + (
                    (r"\s*(" + _base.FrequentUsedPatter.float
                     + ")\s*" + r',\s*') * 2
            ) + "(" + _base.FrequentUsedPatter.float + r')\s*\)',
            linelist[19 - 2])[0][0:-1:3]
        ]

        return field_values, field_ranges, title, t

    def build_x1_x2_meshgrid(self, i):
        linelist, t, title, n_line_x1, n_line_x2, n_line_values, i_x1_start, i_x2_start, i_field_value_start, n_x1, n_x2 = self._block_splitter(
            i)
        x1, x2 = self._get_data(linelist, i_x1_start, n_line_x1), self._get_data(linelist, i_x2_start, n_line_x2)
        x1v, x2v = numpy.meshgrid(x1, x2)
        return x1v, x2v, x1, x2, title, t

    @staticmethod
    def _get_data(linelist, index_of_1st_line, n_lines):
        return numpy.fromstring(''.join(linelist[index_of_1st_line:index_of_1st_line + n_lines]), dtype=float,
                                sep=' ')

    def get_all_field_values(self):
        t0 = time.time()
        field_values_all_t = {}
        # Assume x y data are the same all the time
        for i in range(len(self.block_list)):
            field_values, field_ranges, title, t = self.get_values_by_index(i)
            field_values_of_this_title = field_values_all_t.get(title, [])
            field_values_of_this_title.append({
                "t": t,
                "data": {
                    "x1s": self.x1x2grid[0],
                    "x2s": self.x1x2grid[1],
                    "field_value": field_values.reshape(self.x1x2grid[0].shape),
                    "field_value_range": field_ranges
                }
            })
            field_values_all_t[title] = field_values_of_this_title
        logger.info("%.2f" % (time.time() - t0))
        return field_values_all_t

    def get_field_value_by_time(self, t, title=" FIELD EZ @OSYS$AREA,SHADE-#1"):
        if self.field_values_all_t == None:
            self.field_values_all_t = self.get_all_field_values()
        t_actual, field_value, i = _base.find_data_near_t(
            self.field_values_all_t[title], t,
            lambda field_values_all_t, j: field_values_all_t[j]['t'],
            lambda field_values_all_t, j: field_values_all_t[j]["data"][
                'field_value'])
        return t_actual, field_value, i


if __name__ == '__main__':
    filename = r"F:\MagicFiles\CherenkovAcc\cascade\Coax-2-cascade-higher-gradient-06.fld"
    fld = FLD(filename)
    t, field, i = fld.get_field_value_by_time(1e-12)

    pass
