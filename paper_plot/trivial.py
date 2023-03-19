# -*- coding: utf-8 -*-
# @Time    : 2023/3/15 20:54
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : trivial.py
# @Software: PyCharm
"""
用于论文绘图
"""
import typing

import pandas
from sklearn.cluster import KMeans

import _base
from total_parser import *


def cluster(data: pandas.DataFrame, min_mean_dz=1e-3) -> typing.List[pandas.DataFrame]:
    """
    :param min_mean_dz: 两个聚类中心的最小值，若两个中心距离比此更小，则认为是同一类
    :param data:
    :return: 聚类中心更小的为第0类
    """
    # min_mean_dz = 1e-3  #
    zs = data[0].values.reshape(-1, 1)
    data_range = max(zs) - min(zs)
    if data_range < min_mean_dz:
        logger.info("数据范围(%.2e)小于最小类间距(%.2e)，视为1类" % (data_range, min_mean_dz))
        return [data]
    model = KMeans(2, )
    model.fit(zs)
    if abs(model.cluster_centers_[0] - model.cluster_centers_[1]) < min_mean_dz:
        logger.info("聚类中心距离小于最小类间距，视为1类")
        return [data]
    clustered_data_and_cluster_center: typing.Dict[typing.Tuple[pandas.DataFrame, float]] = {}
    for label in set(model.labels_):
        clustered_data_and_cluster_center[label] = data.iloc[model.labels_ == label, :], model.cluster_centers_[label]
    res = list(clustered_data_and_cluster_center.values())
    res.sort(key=lambda elem: elem[1])
    return [elem[0] for elem in res]


def contour_zoom(t, fld: fld_parser.FLD, par: par_parser.PAR, geom_picture_path, geom_range, contour_title,
                 phasespace_title_z_r,
                 contour_range, ax: plt.Axes, show_particles=False):
    scale_factor = 1e3  # mm
    t_actual, field_data, i = fld.get_field_value_by_time(t, contour_title)

    logger.info("t_actual of Ez: %.4e" % (t_actual))
    vmin, vmax = contour_range

    # logger.info("Start plot")
    # 显示轴对称的另一部分
    x1g, x2g = fld.x1x2grid[contour_title]

    raw_zmin = x1g[0, 0]
    raw_rmin = 0  # x1g[0, 0]  # 0
    raw_zmax = x1g[-1, -1]
    raw_rmax = x2g[-1, -1]

    zmin, rmin, zmax, rmax = geom_range
    z_factors = (numpy.array([zmin, zmax]) - raw_zmin) / (raw_zmax - raw_zmin)
    r_factors = (numpy.array([rmin, rmax]) - raw_rmin) / (raw_rmax - raw_rmin)
    print(z_factors, r_factors)
    print(slice(*((r_factors * x1g.shape[0]).astype(int))), slice(*((z_factors * x1g.shape[1]).astype(int))))
    get_zoomed = lambda arr_2d: arr_2d[
        slice(*((r_factors * x1g.shape[0]).astype(int))), slice(*((z_factors * x1g.shape[1]).astype(int)))]
    zoomed_x1g = get_zoomed(x1g)
    zoomed_x2g = get_zoomed(x2g)

    x1g_sym = numpy.vstack([zoomed_x1g, zoomed_x1g]) * scale_factor
    x2g_sym = numpy.vstack([-zoomed_x2g[::-1], zoomed_x2g]) * scale_factor
    zoomed_field_data = get_zoomed(field_data)

    img_data = mpimg.imread(geom_picture_path)
    img_data_sym = numpy.append(img_data, img_data[::-1], axis=0)

    ax.imshow(img_data_sym,  # alpha=0.7,
              extent=numpy.array((
                  zmin, zmax, -rmax, rmax
              )) * scale_factor)  # 显示几何结构
    cf = ax.contourf(
        x1g_sym, x2g_sym, numpy.vstack([zoomed_field_data[::-1], zoomed_field_data]),
        numpy.linspace(vmin, vmax, 50),
        cmap=plt.get_cmap('jet'),
        alpha=.6, extend='both'
    )

    # cf = axs[0].contourf(*fld.x1x2grid, field_data,cmap=plt.get_cmap('coolwarm'),  # numpy.linspace(-1e6, 1e6, 10)
    #                      )
    # t_actual, phasespace_z_Ek_data, _ = par.get_data_by_time(t_actual, phasespace_title_z_Ek)
    if show_particles:
        t_actual, phase_space_data_z_r, _ = par.get_data_by_time(t_actual, phasespace_title_z_r)
        phase_space_data_z_r = phase_space_data_z_r.values
        phase_space_data_z_r_bottom_side = phase_space_data_z_r.copy()
        phase_space_data_z_r_bottom_side[:, 1] *= -1
        phase_space_data_z_r = numpy.vstack([phase_space_data_z_r, phase_space_data_z_r_bottom_side])
        ax.scatter(*(scale_factor * phase_space_data_z_r).T, c='r', s=.01,
                   )
    ax.set_aspect('equal',  # 'box'
                  )

    ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='x')
    ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    # axs[1].grid()
    ax.set_ylabel("r / mm")
    ax.set_xlabel('z / mm')
    ax.set_title("%2d ps" % (t_actual * 1e12))
    # axs[1].set_ylabel('energy of particle / eV')
    fig = plt.gcf()
    # axs[1].set_title(phasespace_title_z_Ek)
    # pts, labels = ax.get_legend_handles_labels()
    # fig.legend(pts, labels, loc='upper right')
    cbar = fig.colorbar(cf,  # ax=ax,  # location="right"
                        )
    # logger.info("End plot")

    return t_actual


def Ez_vs_zr(t, grd: grd_parser.GRD, par: par_parser.PAR, title_Ez_along_axis, title_phasespace_x1_kE,
             two_axes: typing.Tuple[plt.Axes]):
    titles = [title_Ez_along_axis, title_phasespace_x1_kE]

    # datas = range_datas[titles[0]], phasespace_datas[titles[1]]
    parsers = [grd, par]
    fmts = ["", '.']
    kwargs_for_plot = [dict(), {"markersize": .5, 'color': 'r'}
                       ]
    # ax2 = ax.twinx()
    # axs = [ax, ax2]
    # ylabels = []
    scale_factor = [1e-6, 1e+6]
    datas = []
    for i in range(2):
        # title = titles[i]
        # data_all_time = datas[i]
        # t, data_ = _base.find_data_near_t(data_all_time, t)
        t, data_, _ = parsers[i].get_data_by_time(t, titles[i])
        datas.append(data_)
        two_axes[i].plot(data_.values[:, 0] * 1e3, data_.values[:, 1] * scale_factor[i], fmts[i],
                         label="t = %.2f ps" % (t * 1e12), **kwargs_for_plot[i])
        # axs[i] .set_title(titles[i])
        # axs[i].legend()
    # ax.grid()
    # plt.ticklabel_format(style='sci', scilimits=(-1, 2), axis='x')
    # axs[0].ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    # two_axes[0].set_ylabel("$E_z$ (MV/m)")
    #
    # two_axes[1].set_ylabel(r"r ($\mu m$)")
    zs = datas[1].values[:, 0] * 1e3
    two_axes[1].axvspan(*(zs.min(), zs.max()), alpha=0.3)
    two_axes[0].legend(
        # loc='upper left'
    )
    # plt.gcf().tight_layout()

    # lines, labels = axs[0].get_legend_handles_labels()
    # fig = plt.gcf()
    # fig.legend(lines, labels)


def plot_Ek_along_z(par: par_parser.PAR, Ek_title, t_start, t_end, ax: plt.Axes, do_cluster = True, min_mean_dz=1e-3):
    colors = ['r', 'cornflowerblue']
    for each in par.phasespaces[Ek_title]:
        if each['t'] >= t_start and each['t'] <= t_end:
            datas = cluster(each['data'],min_mean_dz) if do_cluster else  [each['data']]

            for i in range(len(datas)):
                each_bunch = datas[i]
                ax.plot(each_bunch[0] * 1e3, each_bunch[1] / 1e3, '.', markersize=.2,
                        color=colors[i])
    ax.set_ylabel('energy (keV)')
    ax.set_xlabel('z (mm)')
    # plt.gcf().tight_layout()

    ax.grid()



if __name__ == '__main__':
    outdir = r'D:\MagicFiles\tools\pys\paper_plot\.out'

    filename_no_ext = os.path.splitext(
        r"D:\MagicFiles\CherenkovAcc\cascade\251keV-12-add-probe-02-test_bunch_off.grd"
    )[0]
    et = ExtTool(filename_no_ext)
    par = par_parser.PAR(et.get_name_with_ext(ExtTool.FileType.par))
    # a = cluster(par.phasespaces[' ALL PARTICLES @AXES(X1,KE)-#2 $$$PLANE_X1_AND_KE_AT_X0=  0.000'][7]['data'])
    fld = fld_parser.FLD(et.get_name_with_ext(ExtTool.FileType.fld))
    grd = grd_parser.GRD(et.get_name_with_ext(ExtTool.FileType.grd))
    grd.parse_all_observes()

    ts = [4.2e-11, 5.2e-11, 6e-11]
    geom_path = r'D:\MagicFiles\tools\pys\paper_plot\251keV 局部 12E-3 4.6147E-3.png'
    geom_range = [0, 0, 12E-3, 4.6147E-3]

    # 局部放大
    zoomed_geom_path = r'D:\MagicFiles\tools\pys\paper_plot\251keV-add_probe_3.2602e-3 0 11.430e-3 3.0279e-3.png'
    zoomed_geom_range = [3.2602e-3, 0, 11.430e-3, 3.0279e-3]
    # zoomed_geom_path = r'D:\MagicFiles\tools\pys\paper_plot\251keV_6.7828e-3 0 10.530e-3 1.4165e-3.png'
    # zoomed_geom_range = [6.7828e-3 ,0, 10.530e-3 ,1.4165e-3]
    for i in range(len(ts)):
        plt.figure(constrained_layout=True)
        contour_zoom(ts[i], fld, par,
                     geom_picture_path=zoomed_geom_path,
                     geom_range=zoomed_geom_range, contour_title=' FIELD |E| @OSYS$AREA,SHADE-#2',
                     phasespace_title_z_r=" ALL PARTICLES @AXES(X1,X2)-#1 $$$PLANE_X1_AND_X2_AT_X0=  0.000",
                     contour_range=[0, 2.5e8], ax=plt.gca(), show_particles=False)

    # fig, axs = plt.subplots(
    #     2, 1, sharex=True, figsize=(7, 6),
    # )
    fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, constrained_layout=True)
    for i in range(len(ts)):
        two_axes: typing.Tuple[plt.Axes] = (axs[i], axs[i].twinx())
        Ez_vs_zr(ts[i], grd, par, r' FIELD EZ @LINE_AXIS$ #1.1',
                 ' ALL PARTICLES @AXES(X1,X2)-#2 $$$PLANE_X1_AND_X2_AT_X0=  0.000', two_axes)
        two_axes[1].set_ylim(-200, 200)
    two_axes[0].set_xlabel("z (mm)")
    two_axes[0].set_ylabel("$E_z$ (MV/m)")
    two_axes[1].set_ylabel(r"r ($\mu m$)")
    axs[0].set_xlim(3, 12)
    plt.figure()
    plot_Ek_along_z(par, ' ALL PARTICLES @AXES(X1,KE)-#3 $$$PLANE_X1_AND_KE_AT_X0=  0.000', -1, 1000.0, plt.gca(),do_cluster=False)
    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    plot_observe_data(grd, ' FIELD E1 @PROBE0_2,FFT-#13.1', ' FIELD E1 @PROBE0_2,FFT-#13.2', axs)
    axs[1].set_xlim(0, 2e3)
    plt.figure(constrained_layout=True)
    plot_where_is_the_probe(
        zoomed_geom_path, zoomed_geom_range, grd, ' FIELD E1 @PROBE0_2,FFT-#13.1', plt.gca())
    plt.show()
