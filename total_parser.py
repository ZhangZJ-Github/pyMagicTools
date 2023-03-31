# -*- coding: utf-8 -*-
# @Time    : 2023/2/18 19:27
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : total_parser.py
# @Software: PyCharm
import shutil
import time
import typing

import matplotlib

import _base

matplotlib.use("TkAgg")

import os.path
from typing import List

import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # mpimg 用于读取图片
import numpy
import re

import fld_parser
import grd_parser
import par_parser
from _logging import logger
from filenametool import ExtTool, validateTitle
from sympy.physics import units

default_geom_path = r"D:\MagicFiles\CherenkovAcc\cascade\min_case_for_gradient_test\test_diffraction-23.geom.png"


def plot_geom(geom_path: str, geom_range: typing.Iterable[float], ax, alpha=.7, axial_symetry=True):
    """

    :param geom_path:
    :param geom_range:
    :param ax:
    :param alpha:
    :param axial_symetry: if True, 绘制关于x2轴对称的部分
    :return:
    """
    zmin, rmin, zmax, rmax = geom_range

    img_data = mpimg.imread(geom_path)
    if axial_symetry:
        assert rmin == 0
        img_data = numpy.append(img_data, img_data[::-1], axis=0)
        rmin = -rmax

    ax.imshow(img_data, alpha=alpha,
              extent=numpy.array((
                  zmin, zmax, rmin, rmax
              )))  # 显示几何结构


def get_partial_data(geom_range, x1x2grid: typing.Tuple[numpy.ndarray, numpy.ndarray],
                     field_value: typing.List[numpy.ndarray],
                     force_rmin_to_zero=True, n_components=1):
    """
    获取x1x2grid和field_value中位于geom_range中的那部分数据
    :param geom_range:
    :param x1x2grid:
    :param field_value:
    :param force_rmin_to_zero:
    :return:
    """
    logger.info("Start get_partial_data")
    x1g, x2g = x1x2grid
    raw_zmin = x1g[0, 0]
    raw_rmin = 0 if force_rmin_to_zero else x2g[0, 0]
    raw_zmax = x1g[-1, -1]
    raw_rmax = x2g[-1, -1]
    zmin, rmin, zmax, rmax = geom_range
    z_factors = (numpy.array([zmin, zmax]) - raw_zmin) / (raw_zmax - raw_zmin)
    r_factors = (numpy.array([rmin, rmax]) - raw_rmin) / (raw_rmax - raw_rmin)

    get_zoomed_2d = lambda arr_2d: arr_2d[
        slice(*((r_factors * x1g.shape[0] + numpy.array((0, 1))).astype(int))), slice(
            *((z_factors * x1g.shape[1] + numpy.array((0, 1))).astype(int)))]
    zoomed_x1g = get_zoomed_2d(x1g)
    zoomed_x2g = get_zoomed_2d(x2g)
    zoomed_field_data = [None] * 2
    for i in range(n_components):
        zoomed_field_data[i] = get_zoomed_2d(field_value[i])
    return zoomed_x1g, zoomed_x2g, zoomed_field_data


def plot_observe_data(grd, time_domain_data_title, frequency_domain_data_title, axs: typing.Tuple[plt.Axes],
                      semilogy=False):
    td_data, fd_data = grd.obs[time_domain_data_title]['data'].values, grd.obs[frequency_domain_data_title][
        'data'].values
    axs[0].plot(td_data[:, 0] * 1e12, td_data[:, 1] / 1e6)
    axs[0].set_ylabel("$E_z$ (MV/m)")
    axs[0].set_xlabel("t / ps")
    plot_method = axs[1].semilogy if semilogy else axs[1].plot
    plot_method(fd_data[:, 0], fd_data[:, 1] / 1e6)
    axs[1].set_ylabel("Magnitude (MV/m/GHz)")
    axs[1].set_xlabel("frequency / Ghz")
    print(grd.obs[time_domain_data_title]['location_str'])
    print(grd.obs[time_domain_data_title]['describe'])


def plot_where_is_the_probe(geom_path: str, geom_range, grd, title, ax: plt.Axes):
    plot_geom(geom_path, geom_range, ax)  # 显示几何结构
    probe_position = re.findall(_base.FrequentUsedPatter.float + r'(\w+)',
                                grd.obs[title]['location_str'])
    length_unit_to_SI_unit_factors = _base.uc.length_unit_to_SI_unit_factors
    pos = [0, 0]
    for i in range(len(pos)):
        pos[i] = float(''.join(probe_position[i][:2])) * length_unit_to_SI_unit_factors[
            probe_position[i][2]
        ]
    plt.scatter(
        pos[0], pos[1], s=200
    )
    logger.info(pos)


def _par_filtered(particle_data_, particle_frac):
    filter = numpy.random.rand(len(particle_data_)) < particle_frac
    particle_data_ = particle_data_.iloc[filter, :]
    return particle_data_


def plot_Ez_with_phasespace(grd: grd_parser, par: par_parser.PAR, t, axs: List[plt.Axes], particle_frac=0.3,
                            title_Ez_along_axis: str = " FIELD EZ @LINE_AXIS$ #1.1",
                            title_phasespace_x1_kE: str = " ALL PARTICLES @AXES(X1,KE)-#2 $$$PLANE_X1_AND_KE_AT_X0=  0.000"):
    assert len(axs) == 2
    titles = [title_Ez_along_axis, title_phasespace_x1_kE]
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


def get_min_and_max(fld: fld_parser.FLD, contour_title, indexes: slice = None):
    """
    获取.fld的contour数据中的最小值和最大值
    :param fld:
    :param contour_title:
    :return:
    """
    vmax = 0
    vmin = 0
    filtered_generator = fld.all_generator[contour_title]
    if indexes:
        filtered_generator = filtered_generator[indexes]

    for fd in filtered_generator:
        field_range_start, field_range_end, field_range_step = fd['generator'].get_field_range(fld.blocks_groupby_type)
        vmax = max(field_range_end, vmax)
        vmin = min(field_range_start, vmin)
    return vmin, vmax


def plot_contour_vs_phasespace(fld: fld_parser.FLD, par: par_parser.PAR, grd: grd_parser.GRD, t, axs: List[plt.Axes],
                               frac=0.3,
                               geom_picture_path=default_geom_path,
                               geom_range=None,  # zmin, rmin, zmax, rmax
                               old_phasespace_data_z_Ek=numpy.array([[0], [0]]), contour_range=[],
                               contour_title: str = None,
                               phasespace_title_z_Ek: str = " ALL PARTICLES @AXES(X1,KE)-#2 $$$PLANE_X1_AND_KE_AT_X0=  0.000",
                               phasespace_title_z_r: str = " ALL PARTICLES @AXES(X1,X2)-#1 $$$PLANE_X1_AND_X2_AT_X0=  0.000",
                               Ez_title=None, ):
    t_actual, field_data_, i = fld.get_field_value_by_time(t, contour_title)
    field_data = field_data_[0]
    # field_data = generator.get_field_values(fld.blocks_groupby_type)
    logger.info("t_actual of Ez: %.4e" % (t_actual))

    if not contour_range:
        vmin, vmax = get_min_and_max(fld, contour_title)
        _ = max(abs(vmin), vmax)
        vmin, vmax = -_, _
    else:
        vmin, vmax = contour_range

    # logger.info("Start plot")
    # 显示轴对称的另一部分
    x1g, x2g = fld.x1x2grid[contour_title]
    if not geom_range:
        geom_range = [x1g[0, 0], 0, x1g[-1, -1], x2g[-1, -1]]
    zoomed_x1g, zoomed_x2g, zoomed_field_data_ = get_partial_data(
        geom_range, fld.x1x2grid[contour_title], field_data, True)
    zoomed_field_data = zoomed_field_data_[0]
    x1g_sym = numpy.vstack([zoomed_x1g, zoomed_x1g])
    x2g_sym = numpy.vstack([-zoomed_x2g[::-1], zoomed_x2g])

    # field_data = field_data.values

    if (not geom_picture_path) or (not os.path.exists(geom_picture_path)):
        geom_picture_path = default_geom_path
    plot_geom(geom_picture_path, geom_range, axs[0], 1, True)
    logger.info("Geom range %s" % (geom_range))

    cf = axs[0].contourf(
        x1g_sym, x2g_sym, numpy.vstack([zoomed_field_data[::-1], zoomed_field_data]),
        numpy.linspace(vmin, vmax, 50),
        cmap=plt.get_cmap('jet'),
        alpha=.7, extend='both'
    )
    axs[0].set_aspect('equal',  # 'box'
                      )
    # if (not geom_picture_path) or (not os.path.exists(geom_picture_path)):
    #     geom_picture_path = default_geom_path
    #     plot_geom(geom_picture_path, geom_range, axs[0], 1, True)

    # cf = axs[0].contourf(*fld.x1x2grid, field_data,cmap=plt.get_cmap('coolwarm'),  # numpy.linspace(-1e6, 1e6, 10)
    #                      )
    ax_for_Ez: plt.Axes = axs[1].twinx()
    if par is not None:
        t_actual_z_Ek, phasespace_z_Ek_data, _ = par.get_data_by_time(t_actual, phasespace_title_z_Ek)
        t_actual_z_r, phase_space_data_z_r, _ = par.get_data_by_time(t_actual, phasespace_title_z_r)
        logger.info(
            "t_actual of field value: %.4e s, of z-Ek: %.4e s, of z-r: %.4e s" % (
                t_actual, t_actual_z_Ek, t_actual_z_r))
        phase_space_data_z_r = phase_space_data_z_r.values
        phase_space_data_z_r_bottom_side = phase_space_data_z_r.copy()
        phase_space_data_z_r_bottom_side[:, 1] *= -1
        phase_space_data_z_r = numpy.vstack([phase_space_data_z_r, phase_space_data_z_r_bottom_side])
        axs[0].scatter(*phase_space_data_z_r.T, c='w', s=1,
                       )

        # axs[0].set_title(contour_title)
        # fig :plt.Figure= plt.figure()
        # grid_spec = plt.GridSpec(2,10,wspace=0.5,hspace=0.5)
        # axs[0] = fig.add_subplot
        axs[1].scatter(*old_phasespace_data_z_Ek[:, 1:], s=.3,
                       label="old data")

        new_phasespace_z_Ek_data = phasespace_z_Ek_data.values.T
        # for each_phasespace_data_z_Ek in old_phasespace_data_z_Ek:
        axs[1].scatter(*new_phasespace_z_Ek_data, s=.3,
                       label="t = %.4e s" % t_actual)
        old_phasespace_data_z_Ek = numpy.hstack([old_phasespace_data_z_Ek, new_phasespace_z_Ek_data])
        particle_zs = new_phasespace_z_Ek_data[0]
        mid_z = (particle_zs.min() + particle_zs.max()) / 2
        ax_for_Ez.axvline(mid_z, alpha=.3)
        if mid_z > 1e2:
            logger.warning("particle_zs.min(), particle_zs.max() = %.2e,%.2e" % (particle_zs.min(), particle_zs.max()))
    for ax in axs:
        ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='x')
        ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    axs[1].grid()
    axs[0].set_ylabel("r / m")
    axs[1].set_xlabel('z / m')
    axs[1].set_ylabel('energy of particle / eV')

    t_Ez, Ez_data, _ = grd.get_data_by_time(t_actual, Ez_title)

    axs[0].set_xlim(geom_range[0], geom_range[2])
    axs[0].set_ylim(-geom_range[3], geom_range[3])
    ax_for_Ez.plot(*Ez_data.values.T, color='darkred')

    Ez_max_for_plt = max(abs(vmin), abs(vmax))
    ax_for_Ez.set_ylim(ymin=-Ez_max_for_plt, ymax=Ez_max_for_plt)
    ax_for_Ez.set_ylabel("E (V/m)")
    ax_for_Ez.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')

    # axs[1].legend()
    fig = plt.gcf()
    # axs[1].set_title(phasespace_title_z_Ek)
    pts, labels = axs[1].get_legend_handles_labels()
    fig.legend(pts, labels, loc='lower right')
    cbar = fig.colorbar(cf, ax=axs,  # location=""
                        )
    # logger.info("End plot")

    return old_phasespace_data_z_Ek, t_actual


def export_contours_in_folder(
        fld: fld_parser.FLD, par: par_parser.PAR, grd: grd_parser.GRD, et: ExtTool,
        contour_title: str,  # ' FIELD EX @OSYS$AREA,SHADE-#1'
        Ez_title: str, phasespace_title_z_r, phasespace_title_z_Ek,
        res_dir_name,
        t_end, dt=2e-12, contour_range=[]
):
    """
    将contour输出到同一文件夹下
    :param fld:
    :param par:
    :param et:
    :param t_end:
    :param dt:
    :param contour_title:
    :return:
    """

    logger.info('开始输出%s' % fld.filename)
    plt.clf()
    logger.info('plt.clf()')
    # old_phasespace_data = numpy.array([[0], [0]])
    old_pahsespace_data_z_Ek = numpy.array([[0], [0]])
    plt.ioff()
    res_dir_name_with_title = os.path.join(res_dir_name, validateTitle(contour_title))
    os.makedirs(res_dir_name_with_title, exist_ok=True)
    geom_path = et.get_name_with_ext(ExtTool.FileType.geom_png)

    for t in numpy.arange(0, t_end, dt):
        old_pahsespace_data_z_Ek, t_actual = plot_contour_vs_phasespace(
            fld, par, grd, t,
            plt.subplots(
                2, 1, sharex=True,
                figsize=(16, 9), constrained_layout=True)[1],
            frac,
            geom_picture_path=geom_path,
            # geom_range=[-1.3493e-3, 0, 25.257e-3, 3.9963e-3],
            old_phasespace_data_z_Ek=old_pahsespace_data_z_Ek, contour_range=contour_range,
            contour_title=contour_title,
            Ez_title=Ez_title, phasespace_title_z_r=phasespace_title_z_r, phasespace_title_z_Ek=phasespace_title_z_Ek
        )
        plt.gcf().suptitle(os.path.split(res_dir_name)[1])
        # plt.get_current_fig_manager().window.state('zoomed')
        plt.gcf().savefig("%s/%03d_ps.png" % (res_dir_name_with_title, numpy.round(t_actual * 1e12)))
        plt.close(plt.gcf())
    # plt.show()
    logger.info("See result:\n%s" % (res_dir_name_with_title))


def copy_m2d_to_res_folder(res_dir_name, et: ExtTool):
    m2dfn = et.get_name_with_ext(ExtTool.FileType.m2d)
    m2dfn_target = os.path.join(res_dir_name, os.path.split(m2dfn)[1])
    shutil.copyfile(m2dfn, m2dfn_target)


def plot_Ez_z_Ek_all_time(grd, par, ts: typing.Iterable[float], fig_path: str,
                          title_Ez_along_axis: str = " FIELD EZ @LINE_AXIS$ #1.1",
                          title_phasespace_x1_kE: str = " ALL PARTICLES @AXES(X1,KE)-#2 $$$PLANE_X1_AND_KE_AT_X0=  0.000", ):
    t0 = time.time()
    fig_, axs = plt.subplots(2, 1, sharex=True,
                             figsize=(16, 9), constrained_layout=True)
    plt.ioff()
    for t in ts:
        plot_Ez_with_phasespace(grd, par, t, axs, .3, title_Ez_along_axis,
                                title_phasespace_x1_kE=title_phasespace_x1_kE)
    t1 = time.time()
    logger.info("plot用时%.2f" % (t1 - t0))
    fig_.savefig(fig_path)
    t2 = time.time()
    logger.info("Time elapsed for save fig: %.2f s" % (t2 - t1))
    plt.close(fig_)
    logger.info("close用时%.2f" % (time.time() - t2))


def get_sym(x1g_partial, x2g_partial, data_partial: typing.List[numpy.ndarray],
            symmetry_factors):
    assert len(symmetry_factors) == len(data_partial)
    x1g_sym = numpy.vstack([x1g_partial, x1g_partial])
    x2g_sym = numpy.vstack([-x2g_partial[::-1], x2g_partial])
    data_sym: typing.List[numpy.ndarray] = [None] * 2
    for i in range(len(symmetry_factors)):
        data_sym[i] = numpy.vstack([symmetry_factors[i] * data_partial[i][::-1], data_partial[i]])
    return x1g_sym, x2g_sym, data_sym


def plot_vector(t, fld: fld_parser.FLD, par: par_parser.PAR, vector_title, phase_space_zr_title: str, ax: plt.Axes,
                length_unit: units.quantities.Quantity):
    t_actual, data, i_ = fld.get_field_value_by_time(t, vector_title)
    x1g, x2g = fld.x1x2grid[vector_title]
    x1g_partial, x2g_partial, data_partial = x1g, x2g, data  # get_partial_data([6e-3, 0, 10e-3, 0.0005], (x1g, x2g), data, False, 2)
    x1g_sym, x2g_sym, data_sym = get_sym(x1g_partial, x2g_partial, data_partial, [1, -1])

    ax.quiver(x1g_sym / length_unit.scale_factor, x2g_sym / length_unit.scale_factor, *data_sym, data_sym[0],
              cmap=plt.get_cmap('jet')
              # scale =1e-9
              )
    t_zr, zr_data, i_zr = par.get_data_by_time(t, phase_space_zr_title)
    logger.info("t_vector = %.2e, t_zr = %.2e" % (t_actual, t_zr))
    zr_bottom = zr_data.copy()
    zr_bottom[1] *= -1
    zr_data_sym = numpy.vstack([zr_data.values, zr_bottom.values])
    ax.scatter(*(zr_data_sym.T / length_unit.scale_factor),  s=.6,
               c= 'r')
    ax.set_aspect('equal')
    ax.set_xlabel('z / %s' % length_unit.abbrev)
    ax.set_ylabel('r / %s' % length_unit.abbrev)
    ax.set_title("%.2f ps" % (t / 1e-12))


if __name__ == '__main__':
    filename_no_ext = os.path.splitext(
        r"D:\MagicFiles\CherenkovAcc\cascade\min_case_for_gradient_test\test_diffraction-23.tlg"
    )[0]
    phasespace_title_z_Ek = ' ALL PARTICLES @AXES(X1,KE)-#4 $$$PLANE_X1_AND_KE_AT_X0=  0.000'
    phasespace_title_z_r = ' ALL PARTICLES @AXES(X1,X2)-#1 $$$PLANE_X1_AND_X2_AT_X0=  0.000'
    Ez_title = ' FIELD EZ @LINE_AXIS$ #1.1'
    contour_title_Ez = ' FIELD EZ @OSYS$AREA,SHADE-#1'
    contour_title_E_abs = ' FIELD |E| @OSYS$AREA,SHADE-#2'
    obs_title_time_domain = ' FIELD E1 @PROBE0_3,FFT-#7.1'
    obs_title_frequency_domain = ' FIELD E1 @PROBE0_3,FFT-#7.2'

    et = ExtTool(filename_no_ext)
    fld = fld_parser.FLD(et.get_name_with_ext(ExtTool.FileType.fld))
    grd = grd_parser.GRD(et.get_name_with_ext(ExtTool.FileType.grd))
    par = par_parser.PAR(et.get_name_with_ext(ExtTool.FileType.par))
    frac = 1
    res_dir_name = "%s/.out/%s" % (os.path.split(et.filename_no_ext))
    os.makedirs(res_dir_name, exist_ok=True)
    copy_m2d_to_res_folder(res_dir_name, et)
    t_end = grd.ranges[tuple(grd.ranges.keys())[0]][-1]['t']

    # plot_Ez_z_Ek_all_time(grd, par, numpy.arange(0, t_end, 2e-12),
    #                       os.path.join(res_dir_name, '轴上电场.png'),
    #                       Ez_title, phasespace_title_z_Ek)
    #
    # export_contours_in_folder(fld, par, grd, et, contour_title_Ez, Ez_title, phasespace_title_z_r,
    #                           phasespace_title_z_Ek,
    #                           res_dir_name, t_end, 2e-12,
    #                           contour_range=[-1e8, 1e8]
    #                           )
    # _, Ezmax = get_min_and_max(fld, contour_title_E_abs)
    # export_contours_in_folder(fld, par, grd, et, contour_title_E_abs, Ez_title, phasespace_title_z_r,
    #                           phasespace_title_z_Ek,
    #                           res_dir_name, t_end, 2e-12,
    #                           contour_range=[0, Ezmax]
    #                           )
    # Z, R = fld.x1x2grid[contour_title_E_abs]
    # plot_where_is_the_probe(et.get_name_with_ext(et.FileType.geom_png), [Z[0, 0], 0, Z[-1, -1], R[-1, -1]], grd,
    #                         obs_title_time_domain, plt.subplots(constrained_layout=True)[1])
    # plt.gcf().savefig(os.path.join(res_dir_name, 'probe_loc.svg'))
    # plot_observe_data(grd, obs_title_time_domain, obs_title_frequency_domain,
    #                   plt.subplots(2, 1, constrained_layout=True)[1])
    # plt.gcf().savefig(os.path.join(res_dir_name, 'td_fd.svg'))
    phasespace_title_z_r_test_bunch = ' TEST_ELECTRON @AXES(X1,X2)-#2 $$$PLANE_X1_AND_X2_AT_X0=  0.000'
    plot_vector(4.7930e-11, fld, par, ' FIELD E(X1,X2) @CHANNEL1-#1', phasespace_title_z_r_test_bunch,
                plt.subplots(constrained_layout=True)[1], units.mm)

    plt.show()
