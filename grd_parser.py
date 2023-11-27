# -*- coding: utf-8 -*-
# @Time    : 2023/2/17 15:24
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : grd_parser.py
# @Software: PyCharm
"""
用于解析MAGIC产生的grd文件
"""
import enum
import re
import time
import typing
from io import StringIO
from typing import List

import matplotlib
import shapely.geometry

matplotlib.use(
    'tkagg'
)
import matplotlib.pyplot as plt
import numpy
import pandas

import _base
from _logging import logger

frequent_used_unit_to_SI_unit = {
    "time": {
        "ns": 1e-9,
        "ps": 1e-12,
        "fs": 1e-15,
    }
}


class GRD(_base.ParserBase):
    """
    .grd文件的结构：
        几个固定的block（边界，header，grid）
        user_defined_block_1
        user_defined_block_2


    """
    n_block_boundaries = 1
    n_block_header = 1
    n_block_grid = 1

    class Object_:
        class AOTYPE(enum.Enum):
            XCONFORMAL2 = 43
            XLINE = 15
            XFUNCTION = 55
            XMARK = 33
            XPOINT = 13
            XCLINE = 17
            AREA = 6
            OTHER = 10086
            UNKNOWN = 10087

        @staticmethod
        def build_XCONFORMAL2(z1z2r1r2s: numpy.ndarray, ):
            """

            :param z1z2r1r2s: 2D array, shape (1, 4)
            :return:
            """
            z1, z2, r1, r2 = z1z2r1r2s[0]
            return shapely.geometry.box(z1, r1, z2, r2)

        @staticmethod
        def build_XCLINE(z1z2r1r2s: numpy.ndarray, ):
            """

            :param z1z2r1r2s: 2D array, shape (1, 4)
            :return:
            """
            arr = z1z2r1r2s[:, [0, 2, 1, 3]]  # z1 r1 z2 r2
            return shapely.geometry.LineString(arr.reshape((-1, 2)))

        @staticmethod
        def build_XLINE(z1z2r1r2s: numpy.ndarray):
            # arr_ =     numpy.array(z1z2r1r2s)
            arr = z1z2r1r2s[:, [0, 2, 1, 3]]  # z1 r1 z2 r2
            return shapely.geometry.Polygon(arr.reshape((-1, 2)))

        # how_to_build_shape = {
        #     AOTYPE.XCONFORMAL2: build_XCONFORMAL2,
        #     AOTYPE.XLINE: build_XLINE,
        #     AOTYPE.XCLINE: build_XCLINE,
        # }

    def get_block_type(self, block_str) -> _base.ParserBase.BlockType:
        return self.dict_name_to_BlockType.get(re.search(r"\n [a-z,A-Z]+", block_str).group()[2:],
                                               self.BlockType.NOT_IMPLEMENT)

    def __init__(self, filename):
        super(GRD, self).__init__(filename)
        self.parse_all_range_datas()
        self.obs: typing.Dict[str, typing.Dict] = {}
        self.parse_all_observes()

    def parse_geom(self) -> typing.Dict[
        str,
        typing.Union[shapely.geometry.Polygon, shapely.geometry.Point, shapely.geometry.LineString]]:
        how_to_build_shape = {
            GRD.Object_.AOTYPE.XCONFORMAL2: GRD.Object_.build_XCONFORMAL2,
            GRD.Object_.AOTYPE.XLINE: GRD.Object_.build_XLINE,
            GRD.Object_.AOTYPE.XCLINE: GRD.Object_.build_XCLINE,
        }
        self._parse_BOUNDARIES()
        supported_aotype = {GRD.Object_.AOTYPE.XCONFORMAL2, GRD.Object_.AOTYPE.XLINE, GRD.Object_.AOTYPE.XCLINE}
        mask = pandas.Series([False] * self._objects_aobj_aotype_iotype.shape[0])
        for aotype in supported_aotype:
            mask = mask | (self._objects_aobj_aotype_iotype[1] == aotype.name)
        supported_xodata = self._objects_xodata.loc[mask]

        supported_obj_names_indexes_type = {
            name: [
                [], GRD.Object_.AOTYPE.UNKNOWN
            ] for name in
            self._objects_aobj_aotype_iotype[0][mask].unique()
        }
        for i in supported_xodata.index:
            supported_obj_names_indexes_type[self._objects_aobj_aotype_iotype[0][i]][0].append(i)
            supported_obj_names_indexes_type[self._objects_aobj_aotype_iotype[0][i]][1] = GRD.Object_.AOTYPE[
                self._objects_aobj_aotype_iotype[1][i]]
        shapes = {}
        for objname in supported_obj_names_indexes_type:
            shapes[objname] = how_to_build_shape[supported_obj_names_indexes_type[objname][1]](
                self._objects_xodata.loc[supported_obj_names_indexes_type[objname][0]].values[:, 1:5])

        return shapes

    def plot_geom(self, ax: plt.Axes):
        polygons = self.parse_geom()
        for objname in polygons:
            ax.plot(*polygons[objname].exterior.xy, )

    def _parse_BOUNDARIES(self):
        """
        初步解析BOUNDARIES块
        :return:
        """
        geom_data_str = self.blocks_groupby_type[self.BlockType.BOUNDARIES][0]
        geom_data_str_linelist = geom_data_str.splitlines(True)
        objects_array_shape = [int(num) for num in
                               re.findall(r'\s+\$\$\$ARRAY_ \s+([0-9]+)_BY_\s+([0-9]+)_BY_', geom_data_str[:1000])[
                                   0]]  # [列,行]
        i_xodata_start = 16
        i_kodata_start = i_xodata_start + objects_array_shape[1] + 1  # OBJECTS.KODATA(KODIM,KOBJMX)
        i_AOBJ_AOTYPE_IOTYP_start = i_kodata_start + objects_array_shape[1] + 1  # OBJECTS.AOBJ.AOTYPE.IOTYP(KOBJMX
        objects_xodata_str_linelist = geom_data_str_linelist[i_xodata_start:i_xodata_start + objects_array_shape[1]]
        objects_kodata_str_linelist = geom_data_str_linelist[i_kodata_start:i_kodata_start + objects_array_shape[1]]
        objects_AOBJ_AOTYPE_IOTYP_str_linelist = geom_data_str_linelist[
                                                 i_AOBJ_AOTYPE_IOTYP_start:i_AOBJ_AOTYPE_IOTYP_start +
                                                                           objects_array_shape[1]]

        self._objects_xodata = pandas.read_csv(StringIO(''.join(objects_xodata_str_linelist)), sep=r'\s+',
                                               header=None, on_bad_lines='skip')
        self._objects_kodata = pandas.read_csv(StringIO(''.join(objects_kodata_str_linelist)), sep=r'\s+',
                                               header=None, on_bad_lines='skip')
        self._objects_aobj_aotype_iotype = pandas.read_csv(StringIO(''.join(objects_AOBJ_AOTYPE_IOTYP_str_linelist)),
                                                           sep=r'\s+',
                                                           header=None, on_bad_lines='skip')
        self._objects_xodata[0] = self._objects_xodata[0].astype(int)

    def parse_all_observes(self):
        if not self.obs:
            t0 = time.time()
            raw_obs = self.blocks_groupby_type[self.BlockType.OBSERVE]
            self.obs = {}
            for obs_str in raw_obs:
                line_list = obs_str.splitlines(True)
                title = line_list[2][:-1]  # Without '\n'
                # i0 = 131038
                lines_of_data = int(re.findall(r'ARRAY_\s+([0-9]+)_BY_ ', obs_str)[0])
                for i in range(min(100, len(line_list))):  # 查找前几行是否有满足要求的数据开头标记（行数）
                    if re.match(r'\s+' + '%d' % lines_of_data + r'\s*\n', line_list[i]):
                        df = pandas.read_csv(StringIO(''.join(line_list[i + 1:i + 1 + lines_of_data])), sep=r'\s+',
                                             header=None, on_bad_lines='skip')
                        self.obs[title] = {
                            'data': df,
                            'describe': line_list[12][:-1],
                            'location_str': line_list[15][:-1]
                        }
                        break
            logger.info("解析observe data用时： %.2f" % (time.time() - t0))
        return self.obs

    def parse_range_data(self, j):
        """
        解析第j个RANGE数据
        :param j:
        :return:
        """
        rangestr = self.blocks_groupby_type[self.BlockType.RANGE][j]
        # self.ranges = {}

        # for rangestr in raw_ranges:
        line_list = rangestr.splitlines(True)
        title = line_list[2][:-1]  # Without '\n'
        # time_with_unit = re.findall(r" Time [0-9]+.[0-9]+\s+[a-z,A-Z]+", line_list[3040 - 3028])[0][6:].split(" ")
        # t = float(time_with_unit[0]) * frequent_used_unit_to_SI_unit['time'][time_with_unit[1]]
        t = float(''.join(*re.findall(r' RANGE\s+' + _base.FrequentUsedPatter.float, line_list[1])))
        lines_of_data = int(re.findall(r'ARRAY_\s+([0-9]+)_BY_ ', rangestr)[0])
        data = {}
        for i in range(min(100, len(line_list))):  # 查找前几行是否有满足要求的数据开头标记（行数）
            if re.match(r'\s+' + '%d' % lines_of_data + r'\s*\n', line_list[i]):
                df = pandas.read_csv(StringIO(''.join(line_list[i + 1:i + 1 + lines_of_data])), sep=r'\s+',
                                     header=None, on_bad_lines='skip')
                data = {
                    "t": t,
                    "data": df
                }

                break

        return data, title

    def parse_all_range_datas(self):
        t0 = time.time()
        raw_ranges = self.blocks_groupby_type[self.BlockType.RANGE]
        self.ranges = {}
        for i in range(len(raw_ranges)):
            data, title = self.parse_range_data(i)
            range_of_this_title = self.ranges.get(title, [])
            range_of_this_title.append(data)
            self.ranges[title] = range_of_this_title
        logger.debug('解析所有range data: %.2f s' % (time.time() - t0))
        return self.ranges

    def get_data_by_time(self, t, title):
        return _base.find_data_near_t(
            self.ranges[title], t,
            lambda ranges, i: ranges[i]["t"],
            lambda ranges, i: ranges[i]["data"])


def plot_EZ_JZ(all_range_data, t, axs: List[plt.Axes]  # = plt.subplots(2, 1, sharex=True)[1]
               ):
    assert len(axs) == 2
    titles = [" FIELD EZ @LINE_AXIS$ #1.1", " FIELD JZ__ELECTRON @LINE_AXIS$ #2.1"]
    fmts = ["", '']
    # ylabels = []
    for i in range(2):
        title = titles[i]
        data_all_time = all_range_data[title]
        t, data_, _ = _base.find_data_near_t(data_all_time, t)
        # data_.iloc[:, 0] *= 1e3
        axs[i].plot(*(data_.values.T.tolist()), fmts[i], label="t = %.4e" % t)
        axs[i].set_title(titles[i])
        # axs[i].legend()
        axs[i].grid()
    plt.ticklabel_format(style='sci', scilimits=(-1, 2), axis='x')
    axs[1].legend()
    # plt.suptitle("t = %.2e" % t)


if __name__ == '__main__':
    # filename = r"F:\MagicFiles\CherenkovAcc\cascade\Coax-2-cascade-higher-gradient-04.grd"
    filename = r"E:\GeneratorAccelerator\Genac\optmz\Genac10G50keV\粗网格\单独处理\Genac10G50keV2.grd"
    # filename = r"D:\MagicFiles\CherenkovAcc\cascade\min_case_for_gradient_test\test_diffraction-23.grd"

    grd = GRD(filename)
    # grd._parse_BOUNDARIES()
    plt.ion()
    plt.figure()
    grd.plot_geom(plt.gca())
    啊啊啊啊
    grd.get_block_type(grd.block_list[-1])
    rangedata = grd.parse_all_range_datas()
    fig, axs = plt.subplots(2, 1, sharex=True)
    ts = numpy.arange(0e-12, 130e-12, 4e-12)
    # ts =numpy.array([30e-12,40e-12,50e-12,*ts])
    for t in ts:
        plot_EZ_JZ(rangedata, t, axs)

    plt.show()
