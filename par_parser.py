# -*- coding: utf-8 -*-
# @Time    : 2023/2/18 10:38
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : par_parser.py
# @Software: PyCharm
# from enum import Enum
import math
import re
import time
from io import StringIO
from typing import Dict, List, Union

import pandas

import _base
from _logging import logger


class PAR(_base.ParserBase):

    def __init__(self, filename,#n_lines_per_data_group=512# or 99
     ):
        t = time.time()
        super(PAR, self).__init__(filename)
        # self.n_lines_per_data_group = n_lines_per_data_group
        self.parse_all_phase_space_datas(#n_lines_per_data_group
                                         )
        logger.info("PAR(%s)初始化耗时%.2f s" % (filename, time.time() - t))
    @staticmethod
    def get_i_of_data(  lines, i_start=0):
        """
        遇到连续的两行及以上"   0"就认为到达了数据行
        如
        :param lines:
        :param i_start:
        :return:
        """
        i = i_start
        _0_count = 0  # 前面连续几行为0
        for line in lines[i_start:]:
            if line.startswith('   0'):
                _0_count += 1
            else:
                if _0_count >= 2:
                    return i
                _0_count = 0
            i += 1

    def parse_all_phase_space_datas(self,#n_lines_per_data_group=512# or 99
                                    ):
        raw_ranges = self.blocks_groupby_type[self.BlockType.PHASESPACE]
        self.phasespaces: Dict[str, List[Dict[str, Union[float, pandas.DataFrame]]]] = {}  # 所有时间片的相空间数据
        for rangestr in raw_ranges:
            line_list = rangestr.splitlines(True)
            title = line_list[2][:-1]  # Without '\n'

            t = float(''.join(*re.findall(r' PHASESPACE\s+([0-9]+\.[0-9]+)(E[-+][0-9]+)?', line_list[1])))
            lines_of_data = int(re.findall(r"\$\$\$ARRAY_\s+([0-9]+)_BY_ ", line_list[8 - 2])[0])
            i_data_start = self.get_i_of_data(line_list[:100], 10)
            # n_512_lines_block = math.ceil(lines_of_data / n_lines_per_data_group)  # 512行作为1小块，这一变量表示小块的个数
            data_lines = line_list[i_data_start+1 :]
            # for i in range(n_512_lines_block - 1):
            #     data_lines += (line_list[i_data_start + 1:i_data_start + n_lines_per_data_group+1])
            #     i_data_start += n_lines_per_data_group+1
            # data_lines += (line_list[i_data_start + 1:i_data_start + lines_of_data % (n_lines_per_data_group+1) + 1])
            data_str = ''.join(data_lines)

            phasespace_of_this_title = self.phasespaces.get(title, [])
            phasespace_of_this_title.append({
                "t": t,
                "data": pandas.read_csv(StringIO(data_str), sep=r'\s+', header=None,on_bad_lines= 'warn')

            })
            self.phasespaces[title] = phasespace_of_this_title
        return self.phasespaces

    def get_data_by_time(self, t, title):
        """
        :param t:
        :param title:
        :return: t, data, i
        """
        return _base.find_data_near_t(
            self.phasespaces[title], t,
            lambda phase_space_data, i: phase_space_data[i]["t"],
            lambda phase_space_data, i: phase_space_data[i]["data"])


if __name__ == '__main__':
    filename = r"F:\MagicFiles\CherenkovAcc\cascade\Coax-2-cascade-higher-gradient-03.par"
    par = PAR(filename)
    par.parse_all_phase_space_datas()
    pass
