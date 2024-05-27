# -*- coding: utf-8 -*-
"""
@Time ： 2024/2/18 20:42
@Auth ： Zi-Jing Zhang (张子靖)
@File ：GENAC_paperplot.py
@IDE ：PyCharm
"""

import matplotlib

matplotlib.use("TkAgg")
matplotlib.rcParams['font.family'] = 'SimHei'
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
import numpy

import geom_parser
import grd_parser
import fld_parser
import par_parser
from filenametool import ExtTool

if __name__ == '__main__':
    et = ExtTool.from_filename(r"F:\RBWO\结果保存\对比结果保存\加第二个输出\output_20240519_041236_56.m2d")
    filename_no_ext = et.filename_no_ext
    phasespace_title_z_Ek = ' ALL PARTICLES @AXES(X1,KE)-#2 $$$PLANE_X1_AND_KE_AT_X0=  0.000'
    phasespace_title_z_r = ' ALL PARTICLES @AXES(X1,X2)-#1 $$$PLANE_X1_AND_X2_AT_X0=  0.000'
    # Ez_title = ' FIELD EZ @LINE_PARTICLE_MOVING$,FFT #3.1'
    contour_title_Ez = ' FIELD EZ @OSYS$AREA,SHADE-#3'
    # contour_title_E_abs = ' FIELD |E| @OSYS$AREA,SHADE-#3'
    # obs_title_time_domain = ' FIELD E1 @PROBE0_3,FFT-#7.1'
    # obs_title_frequency_domain = ' FIELD E1 @PROBE0_3,FFT-#7.2'

    et = ExtTool(filename_no_ext)
    geom = geom_parser.GEOM(et.get_name_with_ext(ExtTool.FileType.fld))
    geom_parser.export_geometry(geom)
    fld = fld_parser.FLD(et.get_name_with_ext(ExtTool.FileType.fld))
    grd = grd_parser.GRD(et.get_name_with_ext(ExtTool.FileType.grd))
    par = par_parser.PAR(et.get_name_with_ext(ExtTool.FileType.par))

    geom_path, zlim, rlim = geom_parser.export_geometry(geom, True)
    plt.figure()
    imgdata = plt.imread(geom_path)

    x1g, x2g = fld.x1x2grid[contour_title_Ez]
    t_actual,Ezdata_,i = fld .get_field_value_by_time(128841e-12,contour_title_Ez)
    Ezdata = Ezdata_[0]
    contour =plt.contourf(x1g,x2g,Ezdata)
    contour.set_zorder(0)
    img = plt.imshow(imgdata, extent=numpy.array((*zlim, *rlim)))

    img.set_zorder(1)


