# -*- coding: utf-8 -*-
# @Time    : 2023/12/15 0:02
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : mode_check.py
# @Software: PyCharm

import matplotlib

matplotlib.use("TkAgg")
matplotlib.rcParams['font.family'] = 'SimHei'
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
import geom_parser
plt.ion()

geom = geom_parser.GEOM(
    r"E:\GeneratorAccelerator\Genac\optmz\Genac10G50keV-1\RoughMesh\单独处理\test_driver\SWS1-mode2.m2d")
ranges = geom.grd.ranges[' FIELD EZ @LINE_PARTICLE_MOVING$,FFT #3.1']
plt.figure()
axs = [plt.gca(), plt.twinx(plt.gca())]
geom.plot(axs[0])
for rg in ranges[-90:-20]: axs[1].plot(*rg['data'].values.T, label=rg['t']);axs[1].legend()
