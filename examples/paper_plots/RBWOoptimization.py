# -*- coding: utf-8 -*-
# @Time    : 2023/12/3 22:32
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : RBWOoptimization.py
# @Software: PyCharm

import matplotlib

import _base

matplotlib.use("TkAgg")
matplotlib.rcParams['font.family'] = 'SimHei'
import matplotlib.pyplot

matplotlib.pyplot.rcParams['axes.unicode_minus'] = False

from typing import List

import matplotlib.pyplot as plt
from _logging import logger
import fld_parser
import grd_parser
import par_parser
import pandas
def _get_mean(df: pandas.DataFrame, DeltaT):
    """
    获取近周期的时间序列数据df在时间间隔DeltaT内的均值
    :param df: 第0列为时间，第1列为值
    :param DeltaT:
    :return:
    """
    colname_period = 'period'
    df[colname_period] = df[0] // (DeltaT)
    return df.groupby(colname_period).mean().iloc[:-2] # 倒数第二个周期的平均功率。倒数第一个周期可能不全，结果波动很大，故不取。

def plot_EZ_IZ(grd: grd_parser.GRD, Ez_title: str, Iz_title: str, t, axs: List[plt.Axes] = None
               ):
    if not axs:
        plt.figure()
        ax1: plt.Axes = plt.gca()
        axs = [ax1, plt.twinx(ax1)]

    titles = [Ez_title, Iz_title]
    kwargs_for_plot = [{'color': '#1f77b4', 'alpha': 0.1}, {'color': '#d62728', 'alpha': 0.1}]
    factors = [(1, 1), (1, -1)]
    # ylabels = []
    for i in range(2):
        title = titles[i]
        data_all_time = grd.ranges[title]
        t, data_, _ = _base.find_data_near_t(data_all_time, t)
        # data_.iloc[:, 0] *= 1e3
        axs[i].plot(*(data_.values * factors[i]).T, label="t = %.1f ns" % (t / 1e-9), **kwargs_for_plot[i])
        # axs[i].set_title(titles[i])
        # axs[i].legend()
        # axs[i].grid()
    # plt.ticklabel_format(style='sci', scilimits=(-1, 2), axis='x')
    axs[0].xaxis.set_major_formatter(lambda z, pos: '%.0f' % (z / 1e-3))
    axs[0].set_xlabel('z / mm')
    axs[0].yaxis.set_major_formatter(lambda Ez, pos: '%.0f' % (Ez / 1e6))
    axs[0].set_ylabel('$E_z$ / (MV/m)')

    axs[1].yaxis.set_major_formatter(lambda Iz, pos: '%.0f' % (Iz / 1e3))
    axs[1].set_ylabel('$-I_z$ / (kA)')

    # axs[1].legend()
    axs[1].grid(axis='x')

    # plt.suptitle("t = %.2e" % t)


before = 0
after = 1
after_Biased = 2
before_and_after = [before, after, after_Biased]
import geom_parser

geoms = {
    before: geom_parser.GEOM(
        r"F:\papers\OptimizationHPM\ICCEM\support\优化前\RSSE_template_20230628_224345_41810688.grd"),
    after: geom_parser.GEOM(
        r"F:\papers\OptimizationHPM\ICCEM\support\优化后\RSSE_template_20230805_034810_37761024.grd"),
    after_Biased: geom_parser.GEOM(r"F:\papers\OptimizationHPM\ICCEM\support\优化后\base.grd")
}
grds = {key:geoms[key].grd for key in geoms}
pars = {key: par_parser.PAR(geoms[key].filename_no_ext + '.par') for key in geoms}
flds = {key:fld_parser.FLD(geoms[key].filename_no_ext + '.fld') for key in geoms}

# grd_after = grd_parser.GRD(r"F:\papers\OptimizationHPM\ICCEM\Word\support\优化后\RSSE_template_20230805_034810_37761024.grd")
# grd_after = grd_parser.GRD(r"F:\papers\OptimizationHPM\ICCEM\Word\support\优化后\base.grd")
Ez_title = r' FIELD EZ @LINE_PARTICLE_MOVING$,FFT #4.1'
Iz_title = r' FIELD_INTEGRAL J.DA @OSYS$AREA,FFT #5.1'
contour_Ez_title = ' FIELD EZ @OSYS$AREA,SHADE-#2'
z_r_title = r' ALL PARTICLES @AXES(X1,X2)-#1 $$$PLANE_X1_AND_X2_AT_X0=  0.000'
import numpy

i=before

plt.figure(figsize = (3,3))
geoms[i].plot(plt.gca())
t_actual, field_data_, index = flds[i].get_field_value_by_time(14.5e-9,' FIELD EZ @OSYS$AREA,SHADE-#2')
logger.info(t_actual)
plt.contourf(*flds[i].x1x2grid[contour_Ez_title],field_data_[0], numpy.linspace(-50e6, 50e6, 50),
        cmap=plt.get_cmap('jet'),
        alpha=.7, extend='both')
plt.gca().scatter(*pars[i].get_data_by_time(t_actual,z_r_title)[1].values.T,s = 0.0001,color = 'red')
plt.xlim(*flds[i].x1x2grid[contour_Ez_title][0][0,[0,-1]])
plt.ylim(*flds[i].x1x2grid[contour_Ez_title][1][[0,-1],0].ravel())
plt.gca().xaxis.set_major_formatter(lambda z,pos:'%d'%(z/1e-3))
plt.gca().yaxis.set_major_formatter(lambda z,pos:'%d'%(z/1e-3))
plt.savefig('temp.png',dpi = 1000)


grd = grds[after]
plt.figure(figsize=(6, 4))
ax1: plt.Axes = plt.gca()
axs = [ax1, plt.twinx(ax1)]
for range_ in grd.ranges[Ez_title]:

    Ez_title = r' FIELD EZ @LINE_PARTICLE_MOVING$,FFT #4.1'
    Iz_title = r' FIELD_INTEGRAL J.DA @OSYS$AREA,FFT #5.1'
    t = range_['t']
    if t % 0.5e-9 < 0.1e-9:
        plot_EZ_IZ(grd, Ez_title, Iz_title, t, axs)
axs[0].set_ylim(-40e6, 40e6)
axs[1].set_ylim(-35e3, 35e3)
# plt.savefig(r"D:\MagicFiles\tools\pys\examples\paper_plots\.out/%.1fns.png"%(t/1e-9))
# plt.close()
# plt.ion()


import geom_parser, filenametool

geoms = {i: geom_parser.GEOM(grds[i].filename) for i in grds.keys()}
pars = {i: par_parser.PAR(geoms[i].filename_no_ext + filenametool.ExtTool.FileType.par.value) for i in grds.keys()}
flds = {i: fld_parser.FLD(geoms[i].filename_no_ext + filenametool.ExtTool.FileType.fld.value) for i in grds.keys()}

i = before
geom, par, fld = geoms[i], pars[i], flds[i]
t = 15e-9
contour_Ez_title = ' FIELD EZ @OSYS$AREA,SHADE-#2'
z_r_title = r' ALL PARTICLES @AXES(X1,X2)-#1 $$$PLANE_X1_AND_X2_AT_X0=  0.000'
t_actual, field_data_, index = fld.get_field_value_by_time(t, contour_Ez_title)
t_actual, phasespacedata, index = par.get_data_by_time(t, z_r_title)
plt.figure()
geom.plot(plt.gca())
temp_pngname = 'temp.png'
plt.gca().set_ylim(0, None)
plt.axis('off');
plt.savefig(temp_pngname, bbox_inches='tight', pad_inches=0, dpi=200)
xlim, ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
plt.close()

from paper_plot.trivial import *
from total_parser import *


def contour_zoom(t, fld: fld_parser.FLD, par: par_parser.PAR, geom_picture_path, geom_range, contour_title,
                 contour_range, ax: plt.Axes,
                 show_particles=False,
                 phasespace_titles_z_r: typing.List[str] = None,
                 scatter_sizes: typing.List[float] = None):
    """
    绘制局部放大图
    :param t:
    :param fld:
    :param par:
    :param geom_picture_path:
    :param geom_range: [左下x1 左下x2 右上x1 右上x2]
    :param contour_title:
    :param phasespace_title_z_r:
    :param contour_range:
    :param ax:
    :param show_particles:
    :return:
    """
    scale_factor = 1e3  # m to mm
    t_actual, field_data, i = fld.get_field_value_by_time(t, contour_title)

    logger.info("t_actual of Ez: %.4e" % (t_actual))
    vmin, vmax = contour_range
    #
    # # logger.info("Start plot")
    # # 显示轴对称的另一部分
    # x1g, x2g = fld.x1x2grid[contour_title]
    #
    # raw_zmin = x1g[0, 0]
    # raw_rmin = 0  # x1g[0, 0]  # 0
    # raw_zmax = x1g[-1, -1]
    # raw_rmax = x2g[-1, -1]
    #
    # zmin, rmin, zmax, rmax = geom_range
    # z_factors = (numpy.array([zmin, zmax]) - raw_zmin) / (raw_zmax - raw_zmin)
    # r_factors = (numpy.array([rmin, rmax]) - raw_rmin) / (raw_rmax - raw_rmin)
    # print(z_factors, r_factors)
    # print(slice(*((r_factors * x1g.shape[0]).astype(int))), slice(*((z_factors * x1g.shape[1]).astype(int))))
    # get_zoomed = lambda arr_2d: arr_2d[
    #     slice(*((r_factors * x1g.shape[0]).astype(int))), slice(*((z_factors * x1g.shape[1]).astype(int)))]
    # zoomed_x1g = get_zoomed(x1g)
    # zoomed_x2g = get_zoomed(x2g)
    zoomed_x1g, zoomed_x2g, zoomed_field_data_ = get_partial_data(geom_range, fld.x1x2grid[contour_title],
                                                                  field_data, True)

    geom_range = numpy.array(geom_range)  # * scale_factor

    # cf = ax.contourf(
    #     x1g_sym, x2g_sym, numpy.vstack([zoomed_field_data[::-1], zoomed_field_data]),
    #     # numpy.linspace(vmin, vmax, 50),
    #     # cmap=plt.get_cmap('jet'),
    #     # alpha=1, #extend='both'
    # )
    # cf = ax.imshow(numpy.vstack([zoomed_field_data[::-1], zoomed_field_data]))
    # cf = ax.imshow( zoomed_field_data[::-1])
    rs = fld.x1x2grid[contour_title][1][:, 0]
    zs = fld.x1x2grid[contour_title][0][0, :]
    imgdata = plt.imread(geom_picture_path)
    query_pts = numpy.array(numpy.meshgrid(numpy.linspace(*geom_range[[1, 3]], rs.shape[0] // 4),
                                           numpy.linspace(*geom_range[[0, 2]], zs.shape[0] // 4),
                                           indexing='ij')).transpose([1, 2, 0])
    field_value_in_query_pts = scipy.interpolate.interpn((rs, zs), field_data[0], query_pts,
                                                         method='linear', bounds_error=False, fill_value=None, )
    geom_range_sym = geom_range.copy()
    geom_range_sym[1] = - geom_range_sym[3]
    cf = ax.imshow(numpy.vstack((field_value_in_query_pts[::-1], field_value_in_query_pts)),
                   vmin=vmin, vmax=vmax,
                   extent=geom_range_sym[[0, 2, 1, 3]] * scale_factor,
                   cmap=plt.get_cmap('jet'),
                   # alpha = 0.7
                   )

    white = numpy.where(imgdata[:, :, :3] == [1, 1, 1])
    imgdata[white[0], white[1], 3] = 0  # 将白色转化为透明色
    # fuchisa = numpy.where((imgdata[:, :, :3] * 255).astype(int) == [191, 0, 191])
    # imgdata[fuchisa[0], fuchisa[1], 3] = 0.3  # 将洋红色转化为半透明色
    # green = numpy.where((imgdata[:, :, :3] * 255).astype(int) == [0, 191, 0])
    # imgdata[green[0], green[1], :3] = numpy.array([0, 251, 0]) / 255  # 草绿色亮度提高

    ax.imshow(numpy.vstack((imgdata, imgdata[::-1])), alpha=1,
              extent=geom_range_sym[[0, 2, 1, 3]] * scale_factor
              )

    # plt.imsave("test.png",numpy.vstack([zoomed_field_data[::-1], zoomed_field_data]))

    # cf = ax.contourf(
    #     x1g_sym, x2g_sym, numpy.vstack([zoomed_field_data[::-1], zoomed_field_data]),
    #     numpy.linspace(vmin, vmax, 50),
    #     cmap=plt.get_cmap('jet'),
    #     alpha=1, extend='both'
    # )
    # plot_geom(geom_picture_path, geom_range, ax, 0.5, True,)

    # ax.axis('off')

    # cf = axs[0].contourf(*fld.x1x2grid, field_data,cmap=plt.get_cmap('coolwarm'),  # numpy.linspace(-1e6, 1e6, 10)
    #                      )
    # t_actual, phasespace_z_Ek_data, _ = par.get_data_by_time(t_actual, phasespace_title_z_Ek)
    if show_particles:
        # scatter_sizes = [0.001, 0.1]
        # phasespace_titles_z_r = [phasespace_title_z_r_driver, phasespace_title_z_r_witness]
        # alphas = [0.1,1]
        for ii in range(len(phasespace_titles_z_r)):
            phasespace_title_z_r = phasespace_titles_z_r[ii]
            if t_actual > par.phasespaces[phasespace_title_z_r][-1]['t']:
                continue
            t_actual, phase_space_data_z_r, _ = par.get_data_by_time(t_actual, phasespace_title_z_r)
            phase_space_data_z_r = phase_space_data_z_r.values
            phase_space_data_z_r_bottom_side = phase_space_data_z_r.copy()
            phase_space_data_z_r_bottom_side[:, 1] *= -1
            phase_space_data_z_r = numpy.vstack([phase_space_data_z_r, phase_space_data_z_r_bottom_side])
            ax.scatter(*(phase_space_data_z_r).T * scale_factor,
                       c='red',
                       s=scatter_sizes[ii],  # alpha=alphas[i]
                       )
    # ax.set_aspect('equal',  # 'box'
    #               )

    # ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='x')
    # ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    # ax.xaxis.set_major_formatter(lambda z, pos: '%.0f' % (z / 1e-3))
    ax.set_xlabel('z / mm')
    # ax.yaxis.set_major_formatter(lambda r, pos: '%.0f' % (r / 1e-3))
    ax.set_ylabel('r / mm')

    # axs[1].grid()

    ax.set_title("%2.1f ns" % (t_actual * 1e9))
    # axs[1].set_ylabel('energy of particle / eV')
    fig = plt.gcf()
    # axs[1].set_title(phasespace_title_z_Ek)
    # pts, labels = ax.get_legend_handles_labels()
    # fig.legend(pts, labels, loc='upper right')
    # cbar = fig.colorbar(cf,  # ax=ax,  # location="right"
    #                     orientation='horizontal'
    #                     )
    # logger.info("End plot")

    return t_actual


plt.figure()
contour_zoom(t, fld, par, temp_pngname, numpy.array((*xlim, *ylim))[[0, 2, 1, 3]], contour_Ez_title, [-50e6, 50e6],
             plt.gca(), True, [z_r_title], [0.001])
plt.ylim(-32, 32)

plt.figure()
# img_data_ = plt.imread(temp_pngname)
# index_white = numpy.where(img_data_[:,:,:-1] == [1.,1.,1.])
# img_data_[index_white[0],index_white[1],-1] = 0# 白色转透明
# plt.imshow(img_data_,extent=(*xlim,*ylim))
geom.plot(plt.gca())
# plt.contourf(*fld.x1x2grid[contour_Ez_title],field_data_[0], numpy.linspace(-50e6, 50e6, 50),
#         cmap=plt.get_cmap('jet'),
#         alpha=.7, extend='both')
# plt.gca().scatter(*phasespacedata.values.T,s = 0.001,color = 'red')

plt.gca().xaxis.set_major_formatter(lambda z, pos: '%.0f' % (z / 1e-3))
plt.gca().set_xlabel('z / mm')
plt.gca().yaxis.set_major_formatter(lambda r, pos: '%.0f' % (r / 1e-3))
plt.gca().set_ylabel('r / mm')
plt.gca().set_ylim(0, 32e-3)
