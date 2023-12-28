# -*- coding: utf-8 -*-
# @Time    : 2023/11/28 11:49
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : test.py
# @Software: PyCharm


import matplotlib
import numpy

from _logging import logger

matplotlib.use("TkAgg")
matplotlib.rcParams['font.family'] = 'SimHei'
import matplotlib.pyplot

matplotlib.pyplot.rcParams['axes.unicode_minus'] = False

import os.path
import par_parser,grd_parser,geom_parser
import matplotlib.pyplot as plt
par = par_parser.PAR(r"F:\changeworld\HPMCalc\simulation\template\GeneratorAccelerator\Genac10G50keV\Genac10G50keV-2-add_drift2.par")