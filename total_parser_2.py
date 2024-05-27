"""
用于综合展示多种parser的结果
"""
import _base
import geom_parser, filenametool
from filenametool import *
import scipy

# from paper_plot.trivial import *
from total_parser import *



def export_contours_in_folder(
        fld: fld_parser.FLD, par: par_parser.PAR, grd: grd_parser.GRD, et: ExtTool,
        contour_title: str,  # ' FIELD EX @OSYS$AREA,SHADE-#1'
        Ez_title: str, phasespace_title_z_r, phasespace_title_z_Ek,
        res_dir_name=None,
        ts: typing.Iterable[float] = None,
        contour_range: typing.Tuple[float, float] = None,
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
    geom_path = et.get_name_with_ext(ExtTool.FileType.png)
    if res_dir_name is None: res_dir_name = make_default_result_dir(fld)

    for t in ts:
        old_pahsespace_data_z_Ek, t_actual = plot_contour_vs_phasespace(
            fld, par, grd, t,
            plt.subplots(
                2, 1, sharex=True,
                figsize=(2 * 5, 2 * 9), constrained_layout=True)[1],
            frac,
            geom_picture_path=geom_path,
            # geom_range=[-1.3493e-3, 0, 25.257e-3, 3.9963e-3],
            old_phasespace_data_z_Ek=old_pahsespace_data_z_Ek, contour_range=contour_range,
            contour_title=contour_title,
            Ez_title=Ez_title, phasespace_title_z_r=phasespace_title_z_r, phasespace_title_z_Ek=phasespace_title_z_Ek
        )

        plt.gcf().suptitle(None)
        # plt.get_current_fig_manager().window.state('zoomed')
        plt.gcf().savefig("%s/%03d_ps.png" % (res_dir_name_with_title, numpy.round(t_actual * 1e12)))
        plt.close(plt.gcf())
    # plt.show()
    logger.info("See result:\n%s" % (res_dir_name_with_title))


def get_default_result_dir_name(parsed_obj: _base.ParserBase):
    return get_default_result_dir_name_from_filename(parsed_obj.filename)


def make_default_result_dir(parsed_obj: _base.ParserBase):
    res_dir_name = get_default_result_dir_name(parsed_obj)
    os.makedirs(res_dir_name, exist_ok=True)
    return res_dir_name


def get_default_ts(fld: fld_parser.FLD):
    ts = []
    for generator in fld.all_generator[contour_title_Ez]:
        ts.append(generator['t'])
    return ts




if __name__ == '__main__':
    et = ExtTool.from_filename(r"F:\RBWO\TEST\april\降场强1\low1.m2d")
    filename_no_ext = et.filename_no_ext
    phasespace_title_z_Ek = ' ALL PARTICLES @AXES(X1,KE)-#2 $$$PLANE_X1_AND_KE_AT_X0=  0.000'
    phasespace_title_z_r =' ALL PARTICLES @AXES(X1,X2)-#1 $$$PLANE_X1_AND_X2_AT_X0=  0.000'
    Ez_title = ' FIELD EZ @BEAM_LINE$ #1.1'
    contour_title_Ez = ' FIELD EZ @OSYS$AREA,SHADE-#3'
    contour_title_E_abs = ' FIELD |E| @OSYS$AREA,SHADE-#3'
    obs_title_time_domain = ' FIELD E1 @PROBE0_3,FFT-#7.1'
    obs_title_frequency_domain = ' FIELD E1 @PROBE0_3,FFT-#7.2'

    et = ExtTool(filename_no_ext)
    geom = geom_parser.GEOM(et.get_name_with_ext(ExtTool.FileType.fld))
    geom_parser.export_geometry(geom,True)
    fld = fld_parser.FLD(et.get_name_with_ext(ExtTool.FileType.fld))
    grd = grd_parser.GRD(et.get_name_with_ext(ExtTool.FileType.grd))
    par = par_parser.PAR(et.get_name_with_ext(ExtTool.FileType.par))

    frac = 1
    res_dir_name = get_default_result_dir_name(grd)
    os.makedirs(res_dir_name, exist_ok=True)
    copy_m2d_to_result_dir(res_dir_name, et)
    ts = get_default_ts(fld)
    Ezmax = get_EZmax_z_r_t(fld, contour_title_Ez)
    # ts = ts[::2]
    # t_end = grd.ranges[Ez_title][-1]['t']
    # dt = fld.all_generator[contour_title_Ez][1]['t'] - fld.all_generator[contour_title_Ez][0]['t']
    # plot_Ez_z_Ek_all_time(grd, par, numpy.arange(0, t_end, dt),
    #                       os.path.join(res_dir_name, 'Ez_along_z.png'),
    #                       Ez_title, phasespace_title_z_Ek)
    aaaa
    export_contours_in_folder(fld, par, grd, et, contour_title_Ez, Ez_title, phasespace_title_z_r,
                              phasespace_title_z_Ek,
                              res_dir_name, ts,
                              contour_range=[-250e6, 250e6]
                              )
    Ezmax = get_EZmax_z_r_t(fld, contour_title_Ez)

    export_contours_in_folder(fld, par, grd, et, contour_title_E_abs, Ez_title, phasespace_title_z_r,
                              phasespace_title_z_Ek,
                              res_dir_name, ts,
                              contour_range=[0, 20e6]
                              )
