# -*- coding: utf-8 -*-
# @Time    : 2023/2/17 15:24
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : grd_parser.py
# @Software: PyCharm
"""
用于解析MAGIC产生的grd文件
"""
import enum
import re
import time
from io import StringIO
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy
import pandas

import _base

frequent_used_unit_to_SI_unit = {
    "time": {
        "ns": 1e-9,
        "ps": 1e-12,
        "fs": 1e-15,
    }
}


class GRD(_base.ParserBase):
    """
    .grd文件的结构：
        几个固定的block（边界，header，grid）
        user_defined_block_1
        user_defined_block_2


    """
    n_block_boundaries = 1
    n_block_header = 1
    n_block_grid = 1
   

    # class BoundariesBlock :
    #     """
    #     Structure of boundaries block
    #     """
    #     i_arr_info = 7-2
    #     def __init__(self,blklinelist):
    #         self.blklinelist = blklinelist
    #
    #     def parse_arr_info (self):
    #         res =(int(s[:-4]) for s in  re.findall(r"[0-9]+_BY_",self.blklinelist))
    #         n_cols = res[0]
    #         n_values = res[1]
    #         return n_cols, n_values

    def get_block_type(self, block_str) -> _base.ParserBase.BlockType:
        return self.dict_name_to_BlockType.get(re.search(r"\n [a-z,A-Z]+", block_str).group()[2:],
                                               self.BlockType.NOT_IMPLEMENT)

    def __init__(self, filename):
        super(GRD, self).__init__(filename)
        self.parse_all_range_datas()


    def parse_all_range_datas(self):
        raw_ranges = self.blocks_groupby_type[self.BlockType.RANGE]
        self.ranges = {}
        for rangestr in raw_ranges:
            line_list = rangestr.splitlines(True)
            title = line_list[2][:-1]  # Without '\n'
            # time_with_unit = re.findall(r" Time [0-9]+.[0-9]+\s+[a-z,A-Z]+", line_list[3040 - 3028])[0][6:].split(" ")
            # t = float(time_with_unit[0]) * frequent_used_unit_to_SI_unit['time'][time_with_unit[1]]
            t = float(''.join(*re.findall(r' RANGE\s+'+_base.FrequentUsedPatter.float, line_list[1])))
            lines_of_data = int(line_list[3046 - 3028])
            data_str_list = line_list[3047 - 3028:3047 - 3028 + lines_of_data]

            df = pandas.read_csv(StringIO("".join(data_str_list)), sep=r"\s+", header=None)
            range_of_this_title = self.ranges.get(title, [])
            range_of_this_title.append(
                {
                    "t": t,
                    "data": df
                }
            )
            self.ranges[title] = range_of_this_title
        return self.ranges

    def get_data_by_time (self, t, title):
        return _base.find_data_near_t(
            self.ranges[title], t,
            lambda ranges, i: ranges[i]["t"],
            lambda ranges, i: ranges[i]["data"])


def plot_EZ_JZ(all_range_data, t, axs: List[plt.Axes]# = plt.subplots(2, 1, sharex=True)[1]
               ):
    assert len(axs) == 2
    titles = [" FIELD EZ @LINE_AXIS$ #1.1", " FIELD JZ__ELECTRON @LINE_AXIS$ #2.1"]
    fmts = ["", '']
    # ylabels = []
    for i in range(2):
        title = titles[i]
        data_all_time = all_range_data[title]
        t, data_,_ =_base. find_data_near_t(data_all_time, t)
        # data_.iloc[:, 0] *= 1e3
        axs[i].plot(*(data_.values.T.tolist()), fmts[i], label="t = %.4e" % t)
        axs[i].set_title(titles[i])
        # axs[i].legend()
        axs[i].grid()
    plt.ticklabel_format(style='sci', scilimits=(-1, 2), axis='x')
    axs[1].legend()
    # plt.suptitle("t = %.2e" % t)




if __name__ == '__main__':
    filename = r"F:\MagicFiles\CherenkovAcc\cascade\Coax-2-cascade-higher-gradient-04.grd"
    grd = GRD(filename)
    grd.get_block_type(grd.block_list[-1])
    rangedata = grd.parse_all_range_datas()
    fig, axs = plt.subplots(2, 1, sharex=True)
    ts = numpy.arange(0e-12, 130e-12, 4e-12)
    # ts =numpy.array([30e-12,40e-12,50e-12,*ts])
    for t in ts:
        plot_EZ_JZ(rangedata, t, axs)

    plt.show()
