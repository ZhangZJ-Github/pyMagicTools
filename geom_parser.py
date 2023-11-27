# -*- coding: utf-8 -*-
# @Time    : 2023/11/27 13:46
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : geom_parser.py
# @Software: PyCharm
import os.path

import matplotlib

matplotlib.use("TkAgg")
matplotlib.rcParams['font.family'] = 'SimHei'
import matplotlib.pyplot

matplotlib.pyplot.rcParams['axes.unicode_minus'] = False

import os.path

import matplotlib.pyplot as plt

import filenametool
import grd_parser
import log_parser


class Geom:
    def __init__(self, filename):
        """
        :param filename: 后缀无所谓
        """
        self.filename_no_ext = filename_no_ext = os.path.splitext(filename)[0]
        self.grd = grd_parser.GRD(
            filenametool.ExtTool(filename_no_ext).get_name_with_ext(filenametool.ExtTool.FileType.grd))
        self.log = log_parser.LOG(
            filenametool.ExtTool(filename_no_ext).get_name_with_ext(filenametool.ExtTool.FileType.log))

    def plot(self, ax: plt.Axes)->plt.Axes:
        """

        :return: ax
        """
        # TODO
