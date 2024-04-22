# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/17 15:21
@Auth ： Zi-Jing Zhang (张子靖)
@File ：best_result.py
@IDE ：PyCharm
"""
import typing
from _logging import logger

import fld_parser
import geom_parser
import par_parser
import numpy
from filenametool import ExtTool
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

if __name__ == '__main__':
    et = ExtTool.from_filename(r'E:\BigFiles\GENAC\GENACX50kV\optimizing\GenacX50kV_tmplt_20240210_051249_02.m2d')
    fld = fld_parser.FLD(et.get_name_with_ext(ExtTool.FileType.fld))
    par = par_parser.PAR(et.get_name_with_ext(ExtTool.FileType.par))
    geom = geom_parser.GEOM(et.get_name_with_ext(ExtTool.FileType.grd))
    grd = geom.grd
    png_path, zlim, rlim = geom_parser.export_geometry(geom, True)


    contour_Ez_title = r' FIELD EZ @OSYS$AREA,SHADE-#2'
    z_r_title = ' ALL PARTICLES @AXES(X1,X2)-#1 $$$PLANE_X1_AND_X2_AT_X0=  0.000'
    range_Ez_title = ' FIELD EZ @LINE_PARTICLE_MOVING$,FFT #3.1'

    x1g_Ez, x2g_Ez = fld.all_generator[contour_Ez_title][0]['generator'].get_x1x2grid(fld.blocks_groupby_type)
    t = 148.663e-9#135.448E-9
    contour_Ez = fld.get_field_value_by_time(t, contour_Ez_title)[1][0]  # cf_Ez.field_at_time(t)
    vmax = numpy.abs(contour_Ez).max()
    vmin = -vmax
    # fig = plt.figure(figsize=(4,3), constrained_layout =True)
    # gs = plt.GridSpec(2,2,width_ratios=[1, 0.1],)
    # axs =[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[0,1])]
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(4, 3),#width_ratios=[1,0.1],
                            constrained_layout=True
                            )
    axs[0].imshow(plt.imread(et.get_name_with_ext(ExtTool.FileType.png)), extent=[*zlim, *rlim])
    contourf = axs[0].contourf(x1g_Ez, x2g_Ez, contour_Ez, numpy.linspace(vmin, vmax, 21),  # cmap='jet',
                               # cmap = 'plasma',
                               zorder=-1,
                               # extend='both'
                               )
    axs[0].scatter(*par.get_data_by_time(t, z_r_title)[1].values.T, s=0.0002,
                   c='w')
    axs[0].set_ylim(30e-3, None)
    axs[0].yaxis.set_major_formatter(lambda r, pos: '%.0f' % (r / 1e-3))
    axs[0].xaxis.set_major_formatter(lambda r, pos: '%.0f' % (r / 1e-3))
    axs[0].set_ylabel('r / mm')

    color_table = mcolors.TABLEAU_COLORS
    c1, c2 = list(color_table.keys())[:2]

    axs[1].plot(*grd.get_data_by_time(t, range_Ez_title)[1].values.T, c=c1)
    range_JDA_title = ' FIELD_INTEGRAL J.DA @OSYS$AREA,FFT #4.1'
    ax2 = axs[1].twinx()
    I_data = grd.get_data_by_time(t, range_JDA_title)[1]
    ax2.plot(I_data[0], -I_data[1], c=c2)

    axs[1].spines['left'].set_color(c1)
    axs[1].tick_params(axis='y', colors=c1)
    ax2.spines['left'].set_color(c1)

    ax2.spines['right'].set_color(c2)
    ax2.tick_params(axis='y', colors=c2)

    axs[1].set_ylabel('$E_z$ / (MV/m)', c=c1)
    axs[1].yaxis.set_major_formatter(lambda r, pos: '%.1f' % (r / 1e6))

    ax2.set_ylabel('beam current / A', c=c2)

    # fig2 = plt.figure(figsize=(0.2,2),constrained_layout = True)
    cbar = fig.colorbar(contourf,#cax=axs[2],
                          ax=[*axs,  # ax2
                                      ], #location="top",
                        # orientation='horizontal',  # fraction=.1
                        format=lambda r, pos: '%.1f' % (r / 1e6)#,use_gridspec=True
                        )
    # fig2.savefig('TypicalEzContourcolorbar.png',dpi=400)
    # cbar.set_major_formatter(lambda r, pos: '%.1f' % (r / 1e6))
    axs[1].set_xlabel('z / mm')
    fig.savefig('TypicalEzContour.png', dpi=400)



    def _hist_titles(keys: typing.Iterable[str]):
        titles = []
        for key in keys:
            if key.startswith(' HISTOGRAM ELECTRON @ENTIRE VOLUME'):
                titles.append(key)
        return titles

    # t = 148.663E-9
    Ek_hist_titles = _hist_titles(grd.ranges.keys())
    # Ek_hist_titles = [Ek_hist_titles[0]] + Ek_hist_titles[2:]
    labels = ["before SWS1", "@SWS1", "@SWS2", "@SWS3", "after SWS3"]

    fig, axs = plt.subplots(len(labels), 1, sharex=True, figsize=(4, 4),
                            constrained_layout=True
                            )
    for i, ax in enumerate(axs):
        ax: plt.Axes
        # ax.plot(*grd.get_data_by_time(t,Ek_hist_titles[i])[1].values.T)
        _hist_data = grd.get_data_by_time(t, Ek_hist_titles[i])[1]
        if i ==len(labels)-1:
            _hist_data = _hist_data.copy()
            _hist_data[1] +=grd.get_data_by_time(t, Ek_hist_titles[i+1])[1] [1]
        ax.bar(_hist_data[0],
               _hist_data[1] / _hist_data[1].max(),
               width=1e3, label=labels[i])
        ax.legend()
    plt.gca().xaxis.set_major_formatter(lambda r, pos: '%.1f' % (r / 1e3))
    axs[-1].set_xlabel('particle kinetic energy / keV')
    # fig.text(0, 0.5, 'count / arb. unit', va='center', rotation='vertical')
    axs[2].set_ylabel('count / arb. unit')

    plt.savefig('Ek_hist_at_coax_microwave_source.png', dpi=400)
