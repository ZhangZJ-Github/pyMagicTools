# -*- coding: utf-8 -*-
# @Time    : 2023/2/17 15:24
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : grd_parser.py
# @Software: PyCharm
"""
用于解析MAGIC产生的grd文件
"""
import re
import time
import typing
from io import StringIO
from typing import List

import matplotlib.pyplot as plt
import numpy
import pandas

import _base
from _logging import logger

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
        self.obs: typing.Dict[str, typing.Dict] = {}
        self.parse_all_observes()

    def parse_all_observes(self):
        if not self.obs:
            t0 = time.time()
            raw_obs = self.blocks_groupby_type[self.BlockType.OBSERVE]
            self.obs = {}
            for obs_str in raw_obs:
                line_list = obs_str.splitlines(True)
                title = line_list[2][:-1]  # Without '\n'
                # i0 = 131038
                lines_of_data = int(re.findall(r'ARRAY_\s+([0-9]+)_BY_ ', obs_str)[0])
                for i in range(min(100, len(line_list))):  # 查找前几行是否有满足要求的数据开头标记（行数）
                    if re.match(r'\s+' + '%d' % lines_of_data + r'\s*\n', line_list[i]):
                        df = pandas.read_csv(StringIO(''.join(line_list[i + 1:i + 1 + lines_of_data])), sep=r'\s+',
                                             header=None)
                        self.obs[title] = {
                            'data': df,
                            'describe': line_list[12][:-1],
                            'location_str': line_list[15][:-1]
                        }
                        break
            logger.info("解析observe data用时： %.2f" % (time.time() - t0))
        return self.obs

    def parse_range_data(self, j):
        """
        解析第j个RANGE数据
        :param j:
        :return:
        """
        rangestr = self.blocks_groupby_type[self.BlockType.RANGE][j]
        # self.ranges = {}

        # for rangestr in raw_ranges:
        line_list = rangestr.splitlines(True)
        title = line_list[2][:-1]  # Without '\n'
        # time_with_unit = re.findall(r" Time [0-9]+.[0-9]+\s+[a-z,A-Z]+", line_list[3040 - 3028])[0][6:].split(" ")
        # t = float(time_with_unit[0]) * frequent_used_unit_to_SI_unit['time'][time_with_unit[1]]
        t = float(''.join(*re.findall(r' RANGE\s+' + _base.FrequentUsedPatter.float, line_list[1])))
        lines_of_data = int(re.findall(r'ARRAY_\s+([0-9]+)_BY_ ', rangestr)[0])
        data = {}
        for i in range(min(100, len(line_list))):  # 查找前几行是否有满足要求的数据开头标记（行数）
            if re.match(r'\s+' + '%d' % lines_of_data + r'\s*\n', line_list[i]):
                df = pandas.read_csv(StringIO(''.join(line_list[i + 1:i + 1 + lines_of_data])), sep=r'\s+',
                                     header=None)
                data = {
                    "t": t,
                    "data": df
                }

                break

        return data, title

    def parse_all_range_datas(self):
        t0 = time.time()
        raw_ranges = self.blocks_groupby_type[self.BlockType.RANGE]
        self.ranges = {}
        for i in range(len(raw_ranges)):
            data, title = self.parse_range_data(i)
            range_of_this_title = self.ranges.get(title, [])
            range_of_this_title.append(data)
            self.ranges[title] = range_of_this_title
        logger.debug('解析所有range data: %.2f s'%(time.time() - t0 ))
        return self.ranges

    def get_data_by_time(self, t, title):
        return _base.find_data_near_t(
            self.ranges[title], t,
            lambda ranges, i: ranges[i]["t"],
            lambda ranges, i: ranges[i]["data"])


def plot_EZ_JZ(all_range_data, t, axs: List[plt.Axes]  # = plt.subplots(2, 1, sharex=True)[1]
               ):
    assert len(axs) == 2
    titles = [" FIELD EZ @LINE_AXIS$ #1.1", " FIELD JZ__ELECTRON @LINE_AXIS$ #2.1"]
    fmts = ["", '']
    # ylabels = []
    for i in range(2):
        title = titles[i]
        data_all_time = all_range_data[title]
        t, data_, _ = _base.find_data_near_t(data_all_time, t)
        # data_.iloc[:, 0] *= 1e3
        axs[i].plot(*(data_.values.T.tolist()), fmts[i], label="t = %.4e" % t)
        axs[i].set_title(titles[i])
        # axs[i].legend()
        axs[i].grid()
    plt.ticklabel_format(style='sci', scilimits=(-1, 2), axis='x')
    axs[1].legend()
    # plt.suptitle("t = %.2e" % t)


if __name__ == '__main__':
    # filename = r"F:\MagicFiles\CherenkovAcc\cascade\Coax-2-cascade-higher-gradient-04.grd"
    filename = r'F:\Tecent\MyData\WeChat Files\wxid_7252352519612\FileStorage\File\2023-03\ERDUAN(1).grd'

    grd = GRD(filename)
    grd.get_block_type(grd.block_list[-1])
    rangedata = grd.parse_all_range_datas()
    fig, axs = plt.subplots(2, 1, sharex=True)
    ts = numpy.arange(0e-12, 130e-12, 4e-12)
    # ts =numpy.array([30e-12,40e-12,50e-12,*ts])
    for t in ts:
        plot_EZ_JZ(rangedata, t, axs)

    plt.show()
