# -*- coding: utf-8 -*-
# @Time    : 2023/2/19 21:19
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : fld_parser.py
# @Software: PyCharm
import re
import time
import typing

import numpy
import psutil

import _base
from _logging import logger

logger.info("%s importing..." % __file__)


class EzMemManager:
    """
    若释放多次都没有明显降低，说明主要内存占用不在于当前程序，因此需要禁用内存释放功能
    """
    threshold_percent = 95.
    cool_down_time = 10  # 两次释放内存的最小间隔，秒

    def __init__(self):
        self.last_release_time = time.time()

    def release_memory(self, how_to_release: typing.Callable[[], bool]):
        old_percent = psutil.cpu_percent()
        t = time.time()
        if old_percent > self.threshold_percent and t - self.last_release_time > self.cool_down_time:
            how_to_release()
            self.last_release_time = t


class FLD(_base.ParserBase):
    DEFAULT_HEADER_LENGTH = 3000  # 每个block中前几行（包含模拟时间、标题、数据起止行号等摘要信息）的总字符数

    class GeneratorBase:
        def __init__(self, i, t, title, n_line_x1, n_line_x2, n_line_values, i_x1_start, i_x2_start,
                     i_field_value_start, n_x1, n_x2, blocktype: _base.ParserBase.BlockType, n_components):
            self.i, self.t, self.title, self.n_line_x1, self.n_line_x2, self.n_line_values, self.i_x1_start, self.i_x2_start, self.i_field_value_start, self.n_x1, self.n_x2, self.blocktype, self.n_components = i, t, title, n_line_x1, n_line_x2, n_line_values, i_x1_start, i_x2_start, i_field_value_start, n_x1, n_x2, blocktype, n_components
            self._linelist: typing.List[str] = None  # 该block的所有行，而非头几行
            self._x1x2grid: typing.Tuple[numpy.ndarray] = None
            self._field_value: typing.List[numpy.ndarray] = [None] * self.n_components
            # self._field_range: typing.List[float] = None

        def release_memory(self):
            """
            降低内存占用
            :return:
            """
            self._field_value = None

        def _read_line_list(self,
                            blocks_grouped_by_block_type: typing.Dict[_base.ParserBase.BlockType, typing.List[str]]):
            if not self._linelist:
                self._linelist: typing.List[str] = blocks_grouped_by_block_type[self.blocktype][
                    self.i].splitlines(True)

        @staticmethod
        def _get_data(linelist, i_line_start: int, n_lines: int):
            return numpy.fromstring(
                ''.join(linelist[i_line_start:i_line_start + n_lines]),
                dtype=float,
                sep=' ')

        def get_x1x2grid(self,
                         blocks_grouped_by_block_type: typing.Dict[_base.ParserBase.BlockType, typing.List[str]]) -> \
                typing.Tuple[numpy.ndarray, numpy.ndarray]:
            if self._x1x2grid is None:
                self._read_line_list(blocks_grouped_by_block_type)
                x1, x2 = self._get_data(
                    self._linelist, self.i_x1_start, self.n_line_x1
                ), self._get_data(self._linelist, self.i_x2_start, self.n_line_x2)
                self._x1x2grid: typing.Tuple[numpy.ndarray, numpy.ndarray] = numpy.meshgrid(x1, x2)
            return self._x1x2grid

        def get_field_values(self,
                             blocks_grouped_by_block_type: typing.Dict[_base.ParserBase.BlockType, typing.List[str]],
                             ):
            if self._field_value[0] is None:
                self._read_line_list(blocks_grouped_by_block_type)
                for i in range(self.n_components):
                    self._field_value[i] = self._get_data(
                        self._linelist, self.i_field_value_start + i * self.n_line_values, self.n_line_values
                    ).reshape((self.n_x2, self.n_x1))
            return self._field_value

    class ContourGenerator(GeneratorBase):
        def __init__(self, i, t, title, n_line_x1, n_line_x2, n_line_values, i_x1_start, i_x2_start,
                     i_field_value_start, n_x1, n_x2):
            super(FLD.ContourGenerator, self).__init__(i, t, title, n_line_x1, n_line_x2, n_line_values,
                                                       i_x1_start, i_x2_start,
                                                       i_field_value_start, n_x1, n_x2,
                                                       _base.ParserBase.BlockType.SOLIDFILL, 1)
            self._field_range: typing.List[float] = None

        def get_field_range(self,
                            blocks_grouped_by_block_type: typing.Dict[_base.ParserBase.BlockType, typing.List[str]]):
            if self._field_range is None:
                self._read_line_list(blocks_grouped_by_block_type)
                self._field_range = [
                    float(s) for s in re.findall(
                        r'RANGE\(' + (
                                (r"\s*(" + _base.FrequentUsedPatter.float
                                 + ")\s*" + r',\s*') * 2
                        ) + "(" + _base.FrequentUsedPatter.float + r')\s*\)',
                        self._linelist[19 - 2])[0][0:-1:3]
                ]
            return self._field_range

    class VectorGenerator(GeneratorBase):
        def __init__(self, i, t, title, n_line_x1, n_line_x2, n_line_values,
                     i_x1_start, i_x2_start,
                     i_field_value_start, n_x1, n_x2, ):
            super(FLD.VectorGenerator, self).__init__(i, t, title, n_line_x1, n_line_x2, n_line_values,
                                                      i_x1_start, i_x2_start,
                                                      i_field_value_start, n_x1, n_x2,
                                                      _base.ParserBase.BlockType.VECTOR, 2)

    def __init__(self, filename: str):
        logger.info("Start parsing .fld file: %s" % filename)
        t = time.time()

        super(FLD, self).__init__(filename)
        # self.x1x2grid = self.build_x1_x2_meshgrid(0)[:2]
        self.x1x2grid: typing.Dict[str, typing.Tuple[numpy.ndarray, numpy.ndarray]] = {}
        self.all_generator: typing.Dict[
            str, typing.List] = self.get_all_generators()  # 初始时只包含每个block的摘要信息，在需要时可以据此解析完整数据
        self.memory_manager = EzMemManager()
        self.text = None
        logger.info("FLD初始化用时%.2f s" % (time.time() - t))

    def get_time(self, i, blocktype: _base.ParserBase.BlockType):
        """
        获取第i个blocktype类的block的时间
        :param i:
        :return:
        """
        return float(''.join(
            re.findall(r' ((SOLIDFILL)|(VECTOR))\s+' + _base.FrequentUsedPatter.float,
                       self.blocks_groupby_type[blocktype][i][:FLD.DEFAULT_HEADER_LENGTH])[0][3:]))

    def get_block_info(self, i, blocktype: _base.ParserBase.BlockType) -> GeneratorBase:
        header_linelist = self.blocks_groupby_type[blocktype][i][:FLD.DEFAULT_HEADER_LENGTH].splitlines(True)
        t = self.get_time(i, blocktype)
        n_x1, n_x2 = (int(s) for s in re.findall(r'ARRAY_\s+([0-9]+)_BY_\s+([0-9]+)_BY_', header_linelist[8 - 2])[0])
        n_values = n_x1 * n_x2
        n_values_each_line = 5
        n_line_x1, n_line_x2, n_line_values = numpy.ceil(
            numpy.array((n_x1, n_x2, n_values)) / numpy.array(
                (n_values_each_line,) * 3)).astype(
            int)
        title = header_linelist[4 - 2][:-1]  # 不含时间信息的标题
        i_x1_start = 21 - 2
        i_x2_start = i_x1_start + n_line_x1 + 1
        i_field_value_start = i_x2_start + n_line_x2
        if blocktype == _base.ParserBase.BlockType.SOLIDFILL:
            res = FLD.ContourGenerator(i, t, title, n_line_x1, n_line_x2, n_line_values, i_x1_start, i_x2_start,
                                       i_field_value_start, n_x1, n_x2)
        else:
            res = FLD.VectorGenerator(i, t, title, n_line_x1, n_line_x2, n_line_values, i_x1_start, i_x2_start,
                                      i_field_value_start, n_x1, n_x2, )
        return res

    def get_all_generators(self):
        logger.info("Start getting all generators")
        t0 = time.time()
        all_generators = {}
        # Assume x y data are the same all the time
        blktypes = [self.BlockType.SOLIDFILL, self.BlockType.VECTOR]
        for blktype in blktypes:
            for i in range(len(self.blocks_groupby_type[blktype])):
                contour_data = self.get_block_info(i, blktype)
                field_values_of_this_title = all_generators.get(contour_data.title, [])
                if contour_data.title not in self.x1x2grid:
                    self.x1x2grid[contour_data.title] = contour_data.get_x1x2grid(self.blocks_groupby_type)
                try:
                    field_values_of_this_title.append({
                        "t": contour_data.t,
                        "generator": contour_data
                    })
                except ValueError as e:
                    logger.warn("跳过了一个时间片（i = %d, t = %.2e, title = %s）\n%s" % (
                        i, contour_data.t, contour_data.title, e))
                all_generators[contour_data.title] = field_values_of_this_title
        logger.info("获取所有block的基本信息耗时：%.2f" % (time.time() - t0))
        return all_generators

    def get_generator_by_time(self, t, title=" FIELD EZ @OSYS$AREA,SHADE-#1"):
        if self.all_generator == None:
            self.all_generator = self.get_all_generators()
        t_actual, field_value_generator, i = _base.find_data_near_t(
            self.all_generator[title], t,
            lambda generator, j: generator[j]['t'],
            lambda generator, j: generator[j]["generator"])
        return t_actual, field_value_generator, i

    def get_field_value_by_time(self, t, title=" FIELD EZ @OSYS$AREA,SHADE-#1"):
        t_actual, field_value_generator, i = self.get_generator_by_time(t, title)

        def how_to_release_memory():
            for each_title in self.all_generator:
                for j in range(len(self.all_generator[title])):
                    self.all_generator[each_title][j]['generator'].release_memory()
            logger.info("释放了一次内存")
            return True

        self.memory_manager.release_memory(how_to_release_memory)
        return t_actual, field_value_generator.get_field_values(self.blocks_groupby_type), i


if __name__ == '__main__':
    filename = r"D:\MagicFiles\CherenkovAcc\cascade\min_case_for_gradient_test\test_diffraction-23.fld"
    fld = FLD(filename)
    g = fld.get_block_info(0, _base.ParserBase.BlockType.VECTOR)
    mg = g.get_x1x2grid(fld.blocks_groupby_type)
    fd = g.get_field_values(fld.blocks_groupby_type)
    t, field, i = fld.get_field_value_by_time(100e-12, ' FIELD E(X1,X2) @CHANNEL1-#1')

    pass
