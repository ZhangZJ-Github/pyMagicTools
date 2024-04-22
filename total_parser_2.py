"""
用于综合展示多种parser的结果
"""
import matplotlib.pyplot as plt
import numpy
from matplotlib.axis import Axis

import _base
import geom_parser
from filenametool import *
# from paper_plot.trivial import *
from total_parser import *


class MagicResult:
    def __init__(self, grd, fld, par, geom: geom_parser.GEOM, ):
        self.grd = grd
        self.fld = fld

        self.par = par
        self.geom = geom
        # self.png_path, self.xlim,self. ylim = geom_parser.export_geometry(self.geom, True)
        self.temp_initial()
        self.default_size_of_scatter = 0.0002

    def temp_initial(self):

        self.__temp_phasespace_z_r_data = None
        self.__temp_phasespace_z_Ek_data = numpy.array(())  # shape (N,2)

        self.__temp_phasespace_x1g = None
        self.__temp_phasespace_x2g = None

    def plot_geom(self, ax: plt.Axes):
        ax.imshow(plt.imread(self.geom.filename), extent=[*self.geom.x1lim, *self.geom.x2lim])

    def plot_contour(self, contour_title,  # z_r_title, z_Ek_title
                     t,
                     ax: plt.Axes,
                     # x1g_Ez=None, x2g_Ez=None,
                     contour_range: typing.Tuple[float, float] = None,
                     ):
        # if (x1g_Ez is None) or (x2g_Ez is None):
        x1g_Ez, x2g_Ez = fld.x1x2grid[
            contour_title]  # fld.all_generator[contour_title][0]['generator'].get_x1x2grid(fld.blocks_groupby_type)
        # t = 148.663e-9  # 135.448E-9
        contour_Ez = fld.get_field_value_by_time(t, contour_title)[1][0]  # cf_Ez.field_at_time(t)
        if contour_range is None:
            contour_range = (- numpy.abs(contour_Ez).max(), numpy.abs(contour_Ez).max())
        vmin, vmax = contour_range
        # fig = plt.figure(figsize=(4,3), constrained_layout =True)
        # gs = plt.GridSpec(2,2,width_ratios=[1, 0.1],)
        # axs =[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[0,1])]
        # fig, axs = plt.subplots(2, 1, sharex=True, figsize=(4, 3),  # width_ratios=[1,0.1],
        #                         constrained_layout=True
        #                         )
        # self.plot_geom(ax)
        contourf = ax.contourf(x1g_Ez, x2g_Ez, contour_Ez, numpy.linspace(vmin, vmax, 21), cmap='jet',
                               # cmap = 'plasma',
                               zorder=-1,
                               extend='both'
                               )
        return contourf

    def plot_par_data(self, phasespace_title, t, ax,*args,**kwargs) -> numpy.ndarray:
        """

        :param phasespace_title:
        :param t:
        :param ax:
        :return: shape (N,2)
        """
        phasespace_data = self.par.get_data_by_time(t, phasespace_title)[1].values
        ax.scatter(*phasespace_data.T#, s=self.default_size_of_scatter,
                   # c='r'
                   )
        return phasespace_data

    # def set_unit(self, axis: Axis, major_formatter: typing.Callable[[float, typing.Any], str], label: str):
    #     axis.set_major_formatter(major_formatter)
    #     axis.set
    #     axis.set_label(label)

    # def set_unit_mm(self, axis: Axis):
    #     self.set_unit(axis, lambda r, pos: '%.1f' % (r / 1e-3), 'r / mm')
    #
    # def set_unit_MV_m(self, axis: Axis):
    #     self.set_unit(axis, lambda r, pos: '%.1f' % (r / 1e6), '$E_{z}$ / (MV/m)')
    #
    # def set_unit_keV(self, axis: Axis):
    #     self.set_unit(axis, lambda r, pos: '%.1f' % (r / 1e6), 'KE / keV')

    def plot_particle_position(self, phasespace_z_r_title, t, ax: plt.Axes,label ):
        return self.plot_par_data(phasespace_z_r_title, t, ax,label)
        # self.set_unit_mm(ax.xaxis)
        # self.set_unit_mm(ax.yaxis)

    def plot_particle_energy(self, phasespace_z_Ek_title,
                             t, ax: plt.Axes,*arg,**kwargs):
        return self.plot_par_data(phasespace_z_Ek_title, t, ax,*arg,**kwargs)

        # self.set_unit_mm(ax.xaxis)
        # self.set_unit_keV(ax.yaxis)

    def plot_contour_phasespace(self, t, axs: List[plt.Axes],
                                contour_range: typing.Tuple[float, float] = None,
                                contour_title: str = None,
                                phasespace_title_z_Ek: str = " ALL PARTICLES @AXES(X1,KE)-#2 $$$PLANE_X1_AND_KE_AT_X0=  0.000",
                                phasespace_title_z_r: str = " ALL PARTICLES @AXES(X1,X2)-#1 $$$PLANE_X1_AND_X2_AT_X0=  0.000",
                                Ez_title=None, ):
        """

        :param t:
        :param axs:
        # :param old_phasespace_data_z_Ek: shape (N, 2)
        :param contour_range:
        :param contour_title:
        :param phasespace_title_z_Ek:
        :param phasespace_title_z_r:
        :param Ez_title:
        :return:
        """

        self.plot_geom(axs[0])

        cf = self.plot_contour(contour_title, t, axs[0], contour_range)
        phasespace_z_r = self.par.get_data_by_time(t, phasespace_title_z_r)[1].values
        axs[0].scatter(*phasespace_z_r.T, s=self.default_size_of_scatter,
                   c='r'
                       )
        axs[0].xaxis.set_major_formatter(lambda z,pos:"%.1f"%(z/1e-3))
        axs[0].yaxis.set_major_formatter(lambda z,pos:"%.1f"%(z/1e-3))
        axs[0].set_ylabel('r / mm')
        # default_pt_size_z_Ek =0.1
        # if len(self.__temp_phasespace_z_Ek_data):
        #     axs[1].scatter(*self.__temp_phasespace_z_Ek_data.T, s=self.default_size_of_scatter,
        #                    label = 'old data')
        new_z_Ek_data= self.par.get_data_by_time(t, phasespace_title_z_Ek)[1].values

        axs[1].scatter(*new_z_Ek_data.T, s=self.default_size_of_scatter,
                       label='new data',c= 'r')

        axs[1].set_ylim(0, 1e6)
        # self.__temp_phasespace_z_Ek_data = numpy.array(
        #     (*self.__temp_phasespace_z_Ek_data,
        #      *new_z_Ek_data))
        # axs[1].legend()
        cbar = plt.gcf().colorbar(cf,#fraction = 0.05
                                  )
        # cbarax :plt.Axes=cbar.ax
        # cbarax.s



        axs[1].yaxis.set_major_formatter(lambda z,pos:"%.1f"%(z/1e3))
        axs[1].set_ylabel('KE / keV')
        axs[1].set_xlabel('z / mm')


        axs[2].plot(*grd.get_data_by_time(t, Ez_title)[1].values.T)
        axs[2].set_ylim(*contour_range)

        axs[2].yaxis.set_major_formatter(lambda z, pos: "%.1f" % (z / 1e6))
        axs[2].set_ylabel('$E_z$ / (MV/m)')




    def export_contour_phasespace_into_folder(self, ts,
                                              contour_range: typing.Tuple[float, float] = None,
                                              contour_title: str = None,
                                              phasespace_title_z_Ek: str = " ALL PARTICLES @AXES(X1,KE)-#2 $$$PLANE_X1_AND_KE_AT_X0=  0.000",
                                              phasespace_title_z_r: str = " ALL PARTICLES @AXES(X1,X2)-#1 $$$PLANE_X1_AND_X2_AT_X0=  0.000",
                                              Ez_title=None,
                                              res_dir_path: str = None
                                              ):
        if res_dir_path is None: res_dir_path = make_default_result_dir(fld)
        res_dir_name_with_title = os.path.join(res_dir_path, validateTitle(contour_title))
        os.makedirs(res_dir_name_with_title, exist_ok=True)
        # default_time_unit=  1e-9#ns

        for t in ts:
            fig, axs = plt.subplots(2, 1, sharex=True, constrained_layout=True, figsize=(10, 4))
            axs:list
            axs=[*axs,axs[1].twinx()]
            self.plot_contour_phasespace(t, axs, contour_range, contour_title, phasespace_title_z_Ek,
                                         phasespace_title_z_r, Ez_title)
            # axs[0].set_ylim(0, 15e-3)
            fig: plt.Figure
            pngname = os.path.join(res_dir_name_with_title, '%05.5f_ns.png' % (t/1e-9))
            axs[0].set_ylim(0, 30e-3)

            fig.savefig(pngname, dpi=400)

            plt.close(fig)
            logger.info('"%s" saved.' % (pngname))




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
    plt.ioff()
    et = ExtTool.from_filename(
        r"F:\papers\OptimizationHPM\ICCEM\support\优化后\base.m2d")
    filename_no_ext = et.filename_no_ext
    phasespace_title_z_Ek = ' ALL PARTICLES @AXES(X1,KE)-#2 $$$PLANE_X1_AND_KE_AT_X0=  0.000'
    phasespace_title_z_r = ' ALL PARTICLES @AXES(X1,X2)-#1 $$$PLANE_X1_AND_X2_AT_X0=  0.000'
    Ez_title = ' FIELD EZ @LINE_PARTICLE_MOVING$,FFT #4.1'
    contour_title_Ez = ' FIELD EZ @OSYS$AREA,SHADE-#2'
    contour_title_E_abs = ' FIELD |E| @OSYS$AREA,SHADE-#4'
    obs_title_time_domain = ' FIELD E1 @PROBE0_3,FFT-#7.1'
    obs_title_frequency_domain = ' FIELD E1 @PROBE0_3,FFT-#7.2'

    et = ExtTool(filename_no_ext)
    geom = geom_parser.GEOM(et.get_name_with_ext(ExtTool.FileType.fld))
    # png_path, xlim, ylim = geom_parser.export_geometry(geom, True)
    fld = fld_parser.FLD(et.get_name_with_ext(ExtTool.FileType.fld))
    grd = grd_parser.GRD(et.get_name_with_ext(ExtTool.FileType.grd))
    par = par_parser.PAR(et.get_name_with_ext(ExtTool.FileType.par))

    frac = 1
    res_dir_name = get_default_result_dir_name(grd)
    os.makedirs(res_dir_name, exist_ok=True)
    copy_m2d_to_result_dir(res_dir_name, et)
    ts = get_default_ts(fld,contour_title_Ez)
    # ts = ts[::2]
    # t_end = grd.ranges[Ez_title][-1]['t']
    # dt = fld.all_generator[contour_title_Ez][1]['t'] - fld.all_generator[contour_title_Ez][0]['t']
    # plot_Ez_z_Ek_all_time(grd, par, numpy.arange(0, t_end, dt),
    #                       os.path.join(res_dir_name, 'Ez_along_z.png'),
    #                       Ez_title, phasespace_title_z_Ek)
    mr = MagicResult(grd, fld, par, geom)
    # plt.figure()
    # mr.plot_contour(contour_title_Ez,ts[-1],plt.gca(),[-50e6,50e6])
    mr.export_contour_phasespace_into_folder(ts, (-50e6, 50e6), contour_title_Ez, phasespace_title_z_Ek,
                                             phasespace_title_z_r, Ez_title, res_dir_name)
