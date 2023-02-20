# -*- coding: utf-8 -*-
# @Time    : 2023/2/18 19:27
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : total_parser.py
# @Software: PyCharm
import matplotlib

matplotlib.use("TkAgg")

import enum
import os.path
from typing import List

import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # mpimg 用于读取图片
import numpy

import fld_parser
import grd_parser
import par_parser
from _logging import logger


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


def _par_filtered(particle_data_, particle_frac):
    filter = numpy.random.rand(len(particle_data_)) < particle_frac
    particle_data_ = particle_data_.iloc[filter, :]
    return particle_data_


def plot_Ez_with_phasespace(grd: grd_parser, par: par_parser.PAR, t, axs: List[plt.Axes], particle_frac=0.3):
    assert len(axs) == 2
    titles = [" FIELD EZ @LINE_AXIS$ #1.1", " ALL PARTICLES @AXES(X1,KE)-#2 $$$PLANE_X1_AND_KE_AT_X0=  0.000"]
    # datas = range_datas[titles[0]], phasespace_datas[titles[1]]
    parsers = [grd, par]
    fmts = ["", '.']
    kwargs_for_plot = [dict(), dict()  # {"markersize": .2}
                       ]
    # ylabels = []
    for i in range(2):
        title = titles[i]
        # data_all_time = datas[i]
        # t, data_ = _base.find_data_near_t(data_all_time, t)
        t, data_, _ = parsers[i].get_data_by_time(t, titles[i])
        if i == 1:
            if particle_frac < 1.:
                # 筛掉一部分粒子，加快显示速度
                # 已知numpy 1.23.4下面这句会报错
                data_ = _par_filtered(data_, particle_frac)
            # print(len(data_))
        axs[i].plot(*(data_.values.T  # .tolist()
                      ), fmts[i], label="t = %.4e" % t, **kwargs_for_plot[i])
        axs[i].set_title(titles[i])
        # axs[i].legend()
    axs[0].grid()
    plt.ticklabel_format(style='sci', scilimits=(-1, 2), axis='x')
    axs[1].ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    lines, labels = axs[0].get_legend_handles_labels()
    fig = plt.gcf()
    fig.legend(lines, labels
               )

    # axs[1].legend()
    # plt.suptitle("t = %.2e" % t)


def plot_Ez_with_phasespace_by_filename_no_ext(filename_no_ext, t, axs: List[plt.Axes], frac=0.3):
    et = ExtTool(filename_no_ext)
    grd = grd_parser.GRD(et.get_name_with_ext(ExtTool.FileType.grd))
    par = par_parser.PAR(et.get_name_with_ext(ExtTool.FileType.par))
    plot_Ez_with_phasespace(grd, par, t, axs, frac)


def plot_contour_vs_phasespace(fld: fld_parser.FLD, par: par_parser.PAR, t, axs: List[plt.Axes], frac=0.3,
                               geom_picture_path=r"F:\MagicFiles\CherenkovAcc\cascade\Coax-2-cascade-higher-gradient-06.geom.png"):

    contour_title = " FIELD EZ @OSYS$AREA,SHADE-#1"
    t_actual, field_data, i = fld.get_field_value_by_time(t, contour_title)
    field_range_start, field_range_end, field_range_step = fld.field_values_all_t[contour_title][i]['data'][
        'field_value_range']
    vmax = max(field_range_start, field_range_end)
    phasespace_title_z_Ek = " ALL PARTICLES @AXES(X1,KE)-#2 $$$PLANE_X1_AND_KE_AT_X0=  0.000"
    phasespace_title_z_r = " ALL PARTICLES @AXES(X1,X2)-#1 $$$PLANE_X1_AND_X2_AT_X0=  0.000"

    t_actual, phasespace_z_Ek_data, _ = par.get_data_by_time(t_actual, phasespace_title_z_Ek)
    t_actual, phase_space_data_z_r, _ = par.get_data_by_time(t_actual, phasespace_title_z_r)

    logger.info("Start plot")
    # 显示轴对称的另一部分
    x1g, x2g = fld.x1x2grid
    x1g_sym = numpy.vstack([x1g, x1g])
    x2g_sym = numpy.vstack([-x2g[::-1], x2g])
    # field_data = field_data.values
    zmin = x1g[0,0]
    zmax = x1g[-1,-1]
    rmax = x2g[-1,-1]
    axs[0].imshow(mpimg.imread(geom_picture_path),  # alpha=0.7,
                  extent=[
                      zmin, zmax, 0, rmax
                  ])  # 显示几何结构
    cf = axs[0].contourf(
        x1g_sym, x2g_sym, numpy.vstack([field_data[::-1], field_data]),
        numpy.linspace(-vmax, vmax, 10),
        cmap=plt.get_cmap('jet'),
        alpha=.6
    )

    # cf = axs[0].contourf(*fld.x1x2grid, field_data,cmap=plt.get_cmap('coolwarm'),  # numpy.linspace(-1e6, 1e6, 10)
    #                      )
    phase_space_data_z_r = phase_space_data_z_r.values
    phase_space_data_z_r_bottom_side = phase_space_data_z_r.copy()
    phase_space_data_z_r_bottom_side[:, 1] *= -1
    phase_space_data_z_r = numpy.vstack([phase_space_data_z_r, phase_space_data_z_r_bottom_side])
    axs[0].scatter(*phase_space_data_z_r.T, s=1
                   )
    axs[0].set_aspect('equal',  # 'box'
                      )
    # axs[0].set_title(contour_title)
    # fig :plt.Figure= plt.figure()
    # grid_spec = plt.GridSpec(2,10,wspace=0.5,hspace=0.5)
    # axs[0] = fig.add_subplot
    axs[1].scatter(*phasespace_z_Ek_data.values.T, s=.3, label="t = %.4e s" % t_actual)

    plt.ticklabel_format(style='sci', scilimits=(-1, 2), axis='x')
    axs[0].ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    axs[1].ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    axs[1].legend()
    # axs[1].set_title(phasespace_title_z_Ek)
    # pts, labels = axs[1].get_legend_handles_labels()
    # fig.legend(pts, labels)
    cbar = plt.gcf().colorbar(cf, ax=axs,  # location="right"
                        )
    logger.info("End plot")


if __name__ == '__main__':
    plt.ion()
    filename_no_ext = os.path.splitext(r"F:\MagicFiles\CherenkovAcc\cascade\Coax-2-cascade-higher-gradient-06.par")[0]
    et = ExtTool(filename_no_ext)
    fld = fld_parser.FLD(et.get_name_with_ext(ExtTool.FileType.fld))
    grd = grd_parser.GRD(et.get_name_with_ext(ExtTool.FileType.grd))
    par = par_parser.PAR(et.get_name_with_ext(ExtTool.FileType.par))
    frac = 1

    fig, axs = plt.subplots(2, 1, sharex=True)
    for t in numpy.arange(0, 100e-12, 4e-12):
    # for t in [32e-12,36e-12,42e-12,54e-12,64e-12,68e-12,82e-12,108e-12]:
        plot_Ez_with_phasespace(grd , par, t, axs,.3)

    plot_contour_vs_phasespace(fld, par, 10e-12, plt.subplots(2, 1, sharex=True)[1], frac)
    logger.info("Start show")
    plt.show()
    logger.info("End show")
    pass
