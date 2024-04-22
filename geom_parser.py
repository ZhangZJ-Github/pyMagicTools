# -*- coding: utf-8 -*-
# @Time    : 2023/11/27 13:46
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : geom_parser.py
# @Software: PyCharm
import os.path

import matplotlib
import numpy

import _base
from _logging import logger

matplotlib.use("TkAgg")
matplotlib.rcParams['font.family'] = 'SimHei'
import matplotlib.pyplot

matplotlib.pyplot.rcParams['axes.unicode_minus'] = False
import matplotlib.image as mpimg  # mpimg 用于读取图片

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
        # Ended with .png
        self.filename = filenametool.ExtTool(filename_no_ext).get_name_with_ext(filenametool.ExtTool.FileType.png)
        self.grd = grd_parser.GRD(
            filenametool.ExtTool(filename_no_ext).get_name_with_ext(filenametool.ExtTool.FileType.grd))
        self.log = log_parser.LOG(
            filenametool.ExtTool(filename_no_ext).get_name_with_ext(filenametool.ExtTool.FileType.log))
        self._shapes = self.grd.parse_geom()
        # 因log文件中structure generator result表的长度限制，截取每项的前几个字符
        self.shapes = {key[:GEOM.MAX_CHAR_OF_VALUE_IN_STRUCTURE_GENERATOR_RESULT]: self._shapes[key] for key in
                       self._shapes}
        self.export_geometry(True)


    def plot(self, ax: plt.Axes) -> plt.Axes:
        """
        :return: ax
        """
        for i in self.log.geom_structure_generator_result.index:
            objname = self.log.geom_structure_generator_result['ObjectName'][i]
            self.plot_shape(ax, self.shapes.get(objname, objname),
                            ObjectType[self.log.geom_structure_generator_result['Type'][i]])
        return ax


    def export_geometry(geom, white_to_transparent: bool = False):
        """

        @return:  png_path, xlim, ylim
        """
        et = filenametool.ExtTool(geom.filename_no_ext)
        plt.figure(  # tight_layout = True
        )
        geom.plot(plt.gca())
        png_path =geom.filename# et.get_name_with_ext(et.FileType.png)
        # plt.gca().set_ylim(0, None)
        plt.axis('off')
        plt.margins(0, 0)
        # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.savefig(png_path,
                    bbox_inches='tight', pad_inches=0,
                    dpi=200
                    )
        geom.x1lim, geom. x2lim = plt.gca().get_xlim(), plt.gca().get_ylim()
        plt.close()
        if white_to_transparent:
            png_white_to_transparent(png_path)
        return png_path, geom. x1lim, geom. x2lim

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


def save_fig_no_margins():
    pass


def export_geometry(geom: GEOM, white_to_transparent: bool = False):
    """

    @return:  png_path, xlim, ylim
    """
    return geom.export_geometry(white_to_transparent)
def png_color_to_transparent(png_path:str,color:typing.Tuple[float,float,float]):

    """
    将png中的某一种颜色改为透明
    """
    # matplotlib.colors.colorConverter.to_rgb()

    img_data = mpimg.imread(png_path)
    # logger.info(set(['%s' % color for color in ((255 * img_data).astype(int)).reshape(-1, 4).tolist()]))
    rgb_arr = (#255 *
              img_data[:, :, :3]).astype(int)

    flter = numpy.ones_like(True, shape =rgb_arr.shape[:2])
    for i,channel_value in enumerate(color):
        flter&=(rgb_arr[:, :, i] == channel_value)
    _index = numpy.where(flter)

    img_data[(*_index), 3] = 0
    plt.imsave(png_path, img_data)

def png_white_to_transparent(png_path: str):
    """
    将png中白色部分转化为透明色
    """
    png_color_to_transparent(png_path,[1.,1.,1.])


if __name__ == '__main__':
    plt.ion()
    # filename = r"D:\MagicFiles\CherenkovAcc\cascade\min_case_for_gradient_test\test_diffraction-23.grd"
    geom = GEOM(r"E:\BigFiles\GENAC\GENACX50kV\optimizing\GenacX50kV_tmplt_20240210_051249_02.m2d")
    export_geometry(geom, True)

    plt.figure()
    geom.plot(plt.gca())
