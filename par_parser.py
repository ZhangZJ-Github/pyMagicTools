# -*- coding: utf-8 -*-
# @Time    : 2023/2/18 10:38
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : par_parser.py
# @Software: PyCharm
# from enum import Enum
import math
import re
from io import StringIO
from typing import Dict, List, Union

import pandas

import _base


class PAR(_base.ParserBase):

    def __init__(self, filename):
        super(PAR, self).__init__(filename)
        self.parse_all_phase_space_datas()

    def parse_all_phase_space_datas(self):
        raw_ranges = self.blocks_groupby_type[self.BlockType.PHASESPACE]
        self.phasespaces: Dict[str, List[Dict[str, Union[float, pandas.DataFrame]]]] = {}
        for rangestr in raw_ranges:
            line_list = rangestr.splitlines(True)
            title = line_list[2][:-1]  # Without '\n'

            t = float(''.join(*re.findall(r' PHASESPACE\s+([0-9]+\.[0-9]+)(E[-+][0-9]+)?', line_list[1])))
            lines_of_data = int(re.findall(r"\$\$\$ARRAY_\s+([0-9]+)_BY_ ", line_list[8 - 2])[0])
            i_data_start = 19 - 2
            n_512_lines_block = math.ceil(lines_of_data / 512)  # 512行作为1小块，这一变量表示小块的个数
            data_lines = []
            for i in range(n_512_lines_block - 1):
                data_lines += (line_list[i_data_start + 1:i_data_start + 513])
                i_data_start += 513
            data_lines += (line_list[i_data_start+1:i_data_start + lines_of_data % 512 + 1])
            data_str = ''.join(data_lines)

            phasespace_of_this_title = self.phasespaces.get(title, [])
            phasespace_of_this_title.append({
                "t": t,
                "data": pandas.read_csv(StringIO(data_str), sep=r'\s+', header=None)

            })
            self.phasespaces[title] = phasespace_of_this_title
        return self.phasespaces


if __name__ == '__main__':
    filename = r"F:\MagicFiles\CherenkovAcc\cascade\Coax-2-cascade-higher-gradient-03.par"
    par = PAR(filename)
    par.parse_all_phase_space_datas()
    pass
