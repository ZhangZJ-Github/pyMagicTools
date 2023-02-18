# -*- coding: utf-8 -*-
# @Time    : 2023/2/18 19:27
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : total_parser.py
# @Software: PyCharm
import enum
import os.path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy

import _base
import grd_parser
import par_parser


class ExtTool:
    """
    Magic常见结果文件的后缀
    """

    class FileType(enum.Enum):
        par = ".par"
        fld = ".fld"
        grd = ".grd"

    def __init__(self, filename_no_ext):
        self.filename_no_ext = filename_no_ext

    def get_name_with_ext(self, ext: enum.Enum):
        return self.filename_no_ext + ext.value


def plot_Ez_with_phasespace(range_datas: Dict, phasespace_datas: dict, t, axs: List[plt.Axes], particle_frac=0.3):
    assert len(axs) == 2
    titles = [" FIELD EZ @LINE_AXIS$ #1.1", " ALL PARTICLES @AXES(X1,KE)-#2 $$$PLANE_X1_AND_KE_AT_X0=  0.000"]
    datas = range_datas[titles[0]], phasespace_datas[titles[1]]
    fmts = ["", '.']
    kwargs_for_plot = [dict(), {"markersize": .2}]
    # ylabels = []
    for i in range(2):
        title = titles[i]
        data_all_time = datas[i]
        t, data_ = _base.find_data_near_t(data_all_time, t)
        if i == 1:
            pass
            # 筛掉一部分粒子，加快显示速度
            # 已知numpy 1.23.4下面这句会报错
            filter = numpy.random.rand(len(data_)) < particle_frac
            data_ = data_.iloc[filter,:]
            # print(len(data_))
        axs[i].plot(*(data_.values.T#.tolist()
                      ), fmts[i], label="t = %.4e" % t, **kwargs_for_plot[i])
        axs[i].set_title(titles[i])
        # axs[i].legend()
    axs[0].grid()
    plt.ticklabel_format(style='sci', scilimits=(-1, 2), axis='x')
    axs[1].ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    lines,labels =axs[0].get_legend_handles_labels()
    fig = plt.gcf()
    fig.legend(   lines,labels
    )

    # axs[1].legend()
    # plt.suptitle("t = %.2e" % t)


def plot_Ez_with_phasespace_by_filename_no_ext(filename_no_ext, t, axs: List[plt.Axes], frac=0.3):
    et = ExtTool(filename_no_ext)
    grd = grd_parser.GRD(et.get_name_with_ext(ExtTool.FileType.grd))
    par = par_parser.PAR(et.get_name_with_ext(ExtTool.FileType.par))
    plot_Ez_with_phasespace(grd.ranges, par.phasespaces, t, axs, frac)


if __name__ == '__main__':
    filename_no_ext = os.path.splitext(r"F:\MagicFiles\CherenkovAcc\cascade\Coax-2-cascade-higher-gradient-05.par")[0]
    et = ExtTool(filename_no_ext)
    grd = grd_parser.GRD(et.get_name_with_ext(ExtTool.FileType.grd))
    par = par_parser.PAR(et.get_name_with_ext(ExtTool.FileType.par))
    frac = 0.1

    fig, axs = plt.subplots(2, 1, sharex=True)
    for t in numpy.arange(0, 100e-12, 4e-12):
        plot_Ez_with_phasespace(grd.ranges, par.phasespaces, t, axs,1)
    plt.show()
