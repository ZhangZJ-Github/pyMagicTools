# -*- coding: utf-8 -*-
# @Time    : 2023/3/15 20:54
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : trivial.py
# @Software: PyCharm
"""
用于论文绘图
"""
import pandas
import scipy.interpolate
from sklearn.cluster import KMeans

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
                                                         method='slinear', bounds_error=False, fill_value=None, )
    geom_range_sym = geom_range.copy()
    geom_range_sym[1] = - geom_range_sym[3]
    cf = ax.imshow(numpy.vstack((field_value_in_query_pts[::-1], field_value_in_query_pts)),
                   vmin=vmin, vmax=vmax,
                   extent=geom_range_sym[[0, 2, 1, 3]] * scale_factor,
                   cmap=plt.get_cmap('inferno'),
                   # alpha = 0.7
                   )

    white = numpy.where(imgdata[:, :, :3] == [1, 1, 1])
    imgdata[white[0], white[1], 3] = 0  # 将白色转化为透明色
    fuchisa = numpy.where((imgdata[:, :, :3] * 255).astype(int) == [191, 0, 191])
    imgdata[fuchisa[0], fuchisa[1], 3] = 0.3  # 将洋红色转化为半透明色
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
                       c='deepskyblue',
                       s=scatter_sizes[ii],#alpha=alphas[i]
                       )
    ax.set_aspect('equal',  # 'box'
                  )

    ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='x')
    ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    # axs[1].grid()
    ax.set_ylabel("r / mm")
    ax.set_xlabel('z / mm')
    ax.set_title("%02.2f ps" % (t_actual * 1e12))
    # axs[1].set_ylabel('energy of particle / eV')
    fig = plt.gcf()
    # axs[1].set_title(phasespace_title_z_Ek)
    # pts, labels = ax.get_legend_handles_labels()
    # fig.legend(pts, labels, loc='upper right')
    cbar = fig.colorbar(cf,  # ax=ax,  # location="right"
                        orientation='horizontal'
                        )
    # logger.info("End plot")

    return t_actual


def Ez_vs_zr(t, grd: grd_parser.GRD, par: par_parser.PAR, title_Ez_along_axis, title_phasespace_x1_x2,
             two_axes: typing.Tuple[plt.Axes]):
    titles = [title_Ez_along_axis, title_phasespace_x1_x2]

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
        if i == 1:
            data_ = pandas.DataFrame(numpy.vstack([data_, data_ * [1, -1]]))
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
    zs = zs[zs < 300]  # 筛除不合理的坐标
    logger.info(zs.max())
    logger.info(zs.min())
    two_axes[1].axvline((zs.min() + zs.max()) / 2, alpha=0.3)
    two_axes[0].legend(
        # loc='upper left'
    )
    # plt.gcf().tight_layout()

    # lines, labels = axs[0].get_legend_handles_labels()
    # fig = plt.gcf()
    # fig.legend(lines, labels)


def plot_Ek_along_z(par: par_parser.PAR, Ek_title, t_start, t_end, ax: plt.Axes, do_cluster=True, min_mean_dz=1e-3):
    colors = ['r', 'cornflowerblue']
    for each in par.phasespaces[Ek_title]:
        if each['t'] >= t_start and each['t'] <= t_end:
            datas = cluster(each['data'], min_mean_dz) if do_cluster else [each['data']]

            for i in range(len(datas)):
                each_bunch = datas[i]
                ax.plot(each_bunch[0] * 1e3, each_bunch[1] / 1e6, '.', markersize=.1,
                        color=colors[i])
    ax.set_ylabel('electron energy (MeV)')
    ax.set_xlabel('z (mm)')
    # plt.gcf().tight_layout()

    ax.grid()


if __name__ == '__main__':
    outdir = r'.out'

    filename_no_ext = os.path.splitext(
        r"D:\MagicFiles\CherenkovAcc\cascade\min_case_for_gradient_test\test_diffraction-23-for_paper.m2d",
        # r"D:\MagicFiles\CherenkovAcc\cascade\min_case_for_gradient_test\3MeV-02-for_paper.toc"
    )[0]
    phasespace_title_z_Ek = ' TEST_ELECTRON @AXES(X1,KE)-#5 $$$PLANE_X1_AND_KE_AT_X0=  0.000'
    phasespace_title_z_r = ' TEST_ELECTRON @AXES(X1,X2)-#2 $$$PLANE_X1_AND_X2_AT_X0=  0.000'
    et = ExtTool(filename_no_ext)
    par = par_parser.PAR(et.get_name_with_ext(ExtTool.FileType.par))
    # a = cluster(par.phasespaces[' ALL PARTICLES @AXES(X1,KE)-#2 $$$PLANE_X1_AND_KE_AT_X0=  0.000'][7]['data'])
    fld = fld_parser.FLD(et.get_name_with_ext(ExtTool.FileType.fld))
    grd = grd_parser.GRD(et.get_name_with_ext(ExtTool.FileType.grd))
    # grd.parse_all_observes()

    # ts = [1.1383e-10, 1.1983e-10, 1.358e-10, 1.5178e-10]
    # ts = [40e-12, 50e-12, 62e-12]
    ts = [  # 40e-12, 80e-12,
        120e-12, 136e-12, 145e-12,  # 180e-12
    ]
    # ts = [18e-12,36e-12, 54e-12,84e-12]

    # geom_path = r'D:\MagicFiles\tools\pys\paper_plot\251keV 局部 12E-3 4.6147E-3.png'
    # geom_range = [0, 0, 12E-3, 4.6147E-3]

    # 局部放大
    zoomed_geom_path = r"D:\MagicFiles\CherenkovAcc\cascade\min_case_for_gradient_test\test_diffraction-23-for_paper 0, 0, 40e-3, 13.207e-3.geom.png"
    # zoomed_geom_path = r"D:\MagicFiles\CherenkovAcc\cascade\min_case_for_gradient_test\3MeV-02-for_paper 0,0,26.853e-3,6e-3.png"
    zoomed_geom_range = [0, 0, 40e-3, 13.207e-3]  # 左下x1 x2 右上x1 x2
    # zoomed_geom_path = r"D:\MagicFiles\CherenkovAcc\cascade\min_case_for_gradient_test\test_diffraction-23-0,0,38.116e-3,10.625e-3.geom.png"
    # zoomed_geom_range = [0, 0, 38.116e-3, 10.625e-3]
    # zoomed_geom_path = r'D:\MagicFiles\tools\pys\paper_plot\251keV_6.7828e-3 0 10.530e-3 1.4165e-3.png'
    # zoomed_geom_range = [6.7828e-3 ,0, 10.530e-3 ,1.4165e-3]
    aaa
    for i in range(len(ts)):
        plt.figure(constrained_layout=True, figsize=(4, 4), dpi=100)
        contour_zoom(ts[i], fld, par,
                     geom_picture_path=zoomed_geom_path,
                     geom_range=zoomed_geom_range, contour_title=' FIELD |E| @OSYS$AREA,SHADE-#2',
                     contour_range=[0, 10e8], ax=plt.gca(),
                     show_particles=True,
                     phasespace_titles_z_r=[r' ELECTRON @AXES(X1,X2)-#3 $$$PLANE_X1_AND_X2_AT_X0=  0.000',
                                            r' TEST_ELECTRON @AXES(X1,X2)-#2 $$$PLANE_X1_AND_X2_AT_X0=  0.000'
                                            ],
                     scatter_sizes=[0.001,
                                    1
                                    ],
                     )
        plt.savefig(os.path.join(outdir, "%d.svg" % i))
    aaaaaaa
    # fig, axs = plt.subplots(
    #     2, 1, sharex=True, figsize=(7, 6),
    # )
    fig, axs = plt.subplots(len(ts), 1, sharex=True, sharey=True, constrained_layout=True, figsize=(5, 5))
    for i in range(len(ts)):
        two_axes: typing.Tuple[plt.Axes] = (axs[i], axs[i].twinx())
        Ez_vs_zr(ts[i], grd, par, r' FIELD EZ @LINE_AXIS$ #1.1', phasespace_title_z_r, two_axes)
        two_axes[1].set_ylim(-100, 100)
        two_axes[0].set_ylim(-1000, 1000)
        two_axes[0].set_xlabel("z (mm)")
        two_axes[0].set_ylabel("$E_z$ (MV/m)")
        two_axes[1].set_ylabel(r"r ($\mu m$)")
        axs[0].set_xlim(3, 12)
    plt.savefig(os.path.join(outdir, 'Ez_vs_zr.svg'))

    aaaaaaaa
    plt.figure(figsize=(5, 3))
    plot_Ek_along_z(par, phasespace_title_z_Ek, -1, 1000.0, plt.gca(),
                    do_cluster=False)
    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    obs_title_time_domain = ' FIELD E1 @PROBE0_3,FFT-#7.1'
    obs_title_frequency_domain = obs_title_time_domain[:-1] + '2'
    plot_observe_data(grd, obs_title_time_domain, obs_title_frequency_domain, axs)
    axs[1].set_xlim(0, 2e3)
    plt.figure(constrained_layout=True)
    plot_where_is_the_probe(
        zoomed_geom_path, zoomed_geom_range, grd, obs_title_time_domain, plt.gca())
    vector_title = ' FIELD E(X1,X2) @CHANNEL1-#1'
    plot_vector(49e-12, fld, par, vector_title, phasespace_title_z_r, plt.subplots(constrained_layout=True)[1],
                units.mm)
    plt.show()
