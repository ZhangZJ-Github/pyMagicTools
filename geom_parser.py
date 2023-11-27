# -*- coding: utf-8 -*-
# @Time    : 2023/11/27 13:46
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : geom_parser.py
# @Software: PyCharm
import os.path

import matplotlib
import numpy

from _logging import logger

matplotlib.use("TkAgg")
matplotlib.rcParams['font.family'] = 'SimHei'
import matplotlib.pyplot

matplotlib.pyplot.rcParams['axes.unicode_minus'] = False

import os.path

import matplotlib.pyplot as plt

import filenametool
import grd_parser
import log_parser
import typing
import shapely.geometry
import enum


class ObjectType(enum.Enum):
    CONDUCTOR = 0
    DIELECTRIC = 1
    # VOID = 2
    VACUUM = 3
    PORT = 4
    AXIAL = 5


colormap = {
    ObjectType.CONDUCTOR: (190, 190, 190, 1),  # RGBA
    ObjectType.DIELECTRIC: (0, 255, 127, 1),
    ObjectType.VACUUM: (255, 255, 255, 1),
    ObjectType.PORT: (255, 255, 0, 1),
    ObjectType.AXIAL: (0, 0, 255, 1)

}
colormap_normalized = {t: numpy.array(colormap[t]) / (255, 255, 255, 1) for t in colormap}


class GEOM:
    MAX_CHAR_OF_VALUE_IN_STRUCTURE_GENERATOR_RESULT = 16

    def __init__(self, filename: str):
        """
        :param filename: 后缀无所谓
        """
        self.filename_no_ext = filename_no_ext = os.path.splitext(filename)[0]
        self.grd = grd_parser.GRD(
            filenametool.ExtTool(filename_no_ext).get_name_with_ext(filenametool.ExtTool.FileType.grd))
        self.log = log_parser.LOG(
            filenametool.ExtTool(filename_no_ext).get_name_with_ext(filenametool.ExtTool.FileType.log))
        self._shapes = self.grd.parse_geom()
        # 因log文件中structure generator result表的长度限制，截取每项的前几个字符
        self.shapes = {key[:GEOM.MAX_CHAR_OF_VALUE_IN_STRUCTURE_GENERATOR_RESULT]: self._shapes[key] for key in
                       self._shapes}

    def plot(self, ax: plt.Axes) -> plt.Axes:
        """
        :return: ax
        """
        for i in self.log.geom_structure_generator_result.index:
            objname = self.log.geom_structure_generator_result['ObjectName'][i]
            self.plot_shape(ax, self.shapes.get(objname, objname),
                            ObjectType[self.log.geom_structure_generator_result['Type'][i]])
        return ax

    @staticmethod
    def plot_shape(
            ax: plt.Axes,
            shape: typing.Union[shapely.geometry.Polygon,
                                shapely.geometry.Point,
                                shapely.geometry.LineString,
                                str],
            material: ObjectType):
        if isinstance(shape, str):
            logger.warning("Object '%s' is ignored" % shape)
            return ax
        color = colormap_normalized[material]
        if isinstance(shape, shapely.geometry.Point):
            ax.scatter(*shape.xy)
            return ax
        if isinstance(shape, shapely.geometry.LineString):
            ax.plot(*shape.xy, '--', color=color)
            return ax

        ax.fill(*shape.exterior.xy, facecolor=color)
        return ax


if __name__ == '__main__':
    plt.ion()
    # filename = r"D:\MagicFiles\CherenkovAcc\cascade\min_case_for_gradient_test\test_diffraction-23.grd"
    filename = r"E:\GeneratorAccelerator\Genac\optmz\Genac10G50keV\粗网格\单独处理\Genac10G50keV2.grd"
    geom = GEOM(filename)
    plt.figure()
    geom.plot(plt.gca())
