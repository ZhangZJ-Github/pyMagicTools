# -*- coding: utf-8 -*-
# @Time    : 2023/3/28 10:39
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : export_data_operator.py
# @Software: PyCharm
"""
用于修改导出的par和grd
"""
import os.path
import re
import time
from shutil import copyfile

import matplotlib

matplotlib.use('tkagg')
from _logging import logger
import pandas

import matplotlib.pyplot as plt
import numpy

import _base
import filenametool
import grd_parser


def float2str(num, n_eff_digits: int = 7):
    """
    格式化为形如’0.123456E-12‘形式的字符串（以0开头）
    :return:
    """
    normal_str = ("%%.%dE" % (n_eff_digits - 1)) % (num)
    eff_part, pow = normal_str.split('E')
    if int(pow) + 1 < 0:
        pow_part = "E-%02d" % (-int(pow) - 1)
    else:
        pow_part = "E+%02d" % (int(pow) + 1)
    return ('%%.%df' % n_eff_digits) % (float(eff_part) / 10) + pow_part


class ExportedFile(_base.ParserBase):
    DEFAULT_N_CHAR = 400  # 需要修改的信息只位于前DEFAULT_N_CHAR个字符中

    def __init__(self, filename):
        super(ExportedFile, self).__init__(filename)

    def time_shift(self, delta_t, new_filename: str, blocktype: _base.ParserBase.BlockType):
        """
        将文件中的时间信息全部+delta_t，若修改后<0，则删除
        不会改变此对象
        即 新时刻 = 旧时刻 + delta_t
        :param delta_t:
        :return:
        """
        blocks = self.blocks_groupby_type[blocktype].copy()
        N = len(blocks)
        start_i = 0
        j = 0
        for i in range(N):
            block = blocks[i]
            str_to_change = block[:self.DEFAULT_N_CHAR]
            lines = str_to_change.splitlines(True)
            old_t_str = ''.join(re.findall(_base.FrequentUsedPatter.float, lines[1])[0])
            t = float(old_t_str)
            new_t = t + delta_t
            if new_t < 0:
                start_i = i + 1
                continue
            new_index = "%d" % (j + 1)
            N_space = len("%d" % (i + 1)) - len(new_index)
            new_index = " " * N_space + new_index
            lines[1] = lines[1].replace(" %d" % (i + 1), " " + new_index).replace(old_t_str, float2str(new_t, 7))

            # lines[1] = re.sub("%d" % (i + 1), new_index, lines[1], 1).replace(old_t_str, float2str(new_t, 7))

            if (''.join(lines) + block[self.DEFAULT_N_CHAR:]).startswith(r''' $DMP$DMP$DMP$DMPSTARTBLOCK               1      
 FLUX               0.5308520E-10     9434  37
 ELECTRON(J,Z,RHO,PZ,PRHO,PPHI)-EXPORT'''):  # j+1 == 546 and blocktype == _base.ParserBase.BlockType.FLUX:#lines[1] .startswith( ' FLUX               0.3433500E-11 54620E-10     9434  37'):
                raise RuntimeError
            j += 1
            blocks[i] = ''.join(lines) + block[self.DEFAULT_N_CHAR:]
        blocks = blocks[start_i:]
        header = self.text[:self.DEFAULT_N_CHAR].splitlines(True)[0]
        text = header + "".join(blocks)
        with open(new_filename, 'w') as f:
            f.write(text)


class ExportedGRD(ExportedFile, grd_parser.GRD):
    DEFAULT_N_CHAR = 1000

    def __init__(self, filename):
        super(ExportedFile, self).__init__(filename)
        super(grd_parser.GRD, self).__init__(filename, do_not_initialize_again=True)
        self._mean_value_vs_t = {}

    def get_mean_value_vs_t(self, title: str):
        if title not in self._mean_value_vs_t:
            res = numpy.zeros((len(self.ranges[title]), 2))
            for i in range(res.shape[0]):
                res[i, 0] = self.ranges[title][i]['t']
                res[i, 1] = self.ranges[title][i]['data'][1].mean()
            self._mean_value_vs_t[title] = res
        return self._mean_value_vs_t[title]

    def plot_mean_value_vs_t(self, title, ax: plt.Axes):
        data = self.get_mean_value_vs_t(title)
        ax.plot(*(data.T))

    def scale_then_write(self, scale, new_filename: str):
        t = time.time()
        logger.info("开始scale_then_write")
        new_str = self.text[:self.DEFAULT_N_CHAR].splitlines(True)[
            0]  # "!DMP FORMAT: ASCII PRECISION:SINGLE TYPE:GRD                                    "
        titles = tuple(self.ranges.keys())
        # 假设每个时间片的排列顺序都是titles[0], titles[1]
        N = len(self.ranges[titles[0]])
        block_index = 0
        for i in range(N):
            for j in range(len(titles)):  # 0,1
                title = titles[j]
                new_str += ''.join(
                    self.blocks_groupby_type[self.BlockType.RANGE][block_index][:self.DEFAULT_N_CHAR].splitlines(True)[
                    :25 - 2])
                datas = self.ranges[title]
                new_data: pandas.DataFrame = datas[i]['data'].copy()
                new_data[1] *= scale
                new_data .insert(0, 'nullstr', "  ")
                new_str += (new_data.to_string(index=False, header=False,
                                               formatters=[lambda nullstr : nullstr,
                                                   lambda num: float2str(num, 6),
                                                   lambda num: float2str(num,6)],col_space = 1) + '\n')
                block_index += 1
        with open(new_filename, 'w') as f:
            f.write(new_str)
        logger.info("scale_then_write done, time elapsed: %.2f s" % (time.time() - t))


if __name__ == '__main__':
    filename = r"D:\MagicFiles\CherenkovAcc\cascade\min_case_for_gradient_test\test_export_and_import\test_export2_.grd"
    filename_no_ext = os.path.splitext(filename)[0]
    et = filenametool.ExtTool(filename_no_ext)
    grd = ExportedFile(filename)
    par = ExportedFile(et.get_name_with_ext(et.FileType.par))
    suffix = "_-trim"
    delta_t = -40e-12  # - 4.8000000e-15 + 0.6300000E-14
    trimed_filename_no_ext = filename_no_ext + suffix
    trimed_et = filenametool.ExtTool(trimed_filename_no_ext)
    grd.time_shift(delta_t, trimed_filename_no_ext + ".grd", grd.BlockType.RANGE)
    par.time_shift(delta_t, trimed_filename_no_ext + ".par", grd.BlockType.FLUX)

    trim_grd = ExportedGRD(trimed_et.get_name_with_ext(trimed_et.FileType.grd))
    scale_suffix = '-scaled'
    trim_grd.scale_then_write(2.0, trimed_filename_no_ext + scale_suffix + '.grd')

    copyfile(trimed_et.get_name_with_ext(trimed_et.FileType.par), trimed_filename_no_ext + scale_suffix + '.par')
    # trim_grd.plot_mean_value_vs_t(' EXPORT E2  ERHO(RHO) #6.1-EXPORT', plt.subplots()[1])

    plt.show()
    pass

    # grd.time_shift(-220e-12, filename_no_ext + "_-220ps.grd")
