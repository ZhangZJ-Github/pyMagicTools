# -*- coding: utf-8 -*-
# @Time    : 2023/2/18 15:10
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : base.py
# @Software: PyCharm
import enum
import re
import time
from typing import Dict, List

import numpy


class FrequentUsedPatter:
    # 浮点数，形如+1.5545E+9 1.0
    float = r'([-+]?[0-9]+\.?[0-9]+)([Ee][-+]?[0-9]+)?'


frequent_used_unit_to_SI_unit = {
    "time": {
        "ns": 1e-9,
        "ps": 1e-12,
        "fs": 1e-15,
    }
}


class ParserBase:
    """
    .grd文件的结构：
        几个固定的block（边界，header，grid）
        user_defined_block_1
        user_defined_block_2


    """
    n_file_info = 1  # 用于描述整个文件的编码方式的行数
    n_lines_DMPSTARTBLOCK_tag_each_block = 1

    startblock_tag_pattern = r" \$DMP\$DMP\$DMP\$DMPSTARTBLOCK\s+[0-9]+\s+\n"

    class BlockType(enum.Enum):
        BOUNDARIES = 0
        HEADER = 1
        GRID = 2
        RANGE = 3
        OBSERVE = 4
        PHASESPACE = 5
        SOLIDFILL = 6
        VECTOR = 7
        NOT_IMPLEMENT = 1008611

    dict_name_to_BlockType: Dict[str, BlockType] = {t.name: t for t in BlockType}

    # class BoundariesBlock :
    #     """
    #     Structure of boundaries block
    #     """
    #     i_arr_info = 7-2
    #     def __init__(self,blklinelist):
    #         self.blklinelist = blklinelist
    #
    #     def parse_arr_info (self):
    #         res =(int(s[:-4]) for s in  re.findall(r"[0-9]+_BY_",self.blklinelist))
    #         n_cols = res[0]
    #         n_values = res[1]
    #         return n_cols, n_values

    def get_block_type(self, block_str) -> BlockType:
        return self.dict_name_to_BlockType.get(re.search(r"\n [a-z,A-Z]+", block_str).group()[2:],
                                               self.BlockType.NOT_IMPLEMENT)

    def __init__(self, filename):
        t = time.time()
        self.filename = filename
        self.text = ""
        with open(self.filename, 'r') as f:
            self.text = f.read()
        self.to_blocks()

    def to_blocks(self):
        self.block_index_list = []
        # for blktype_name in self.dict_name_to_BlockType.keys():
        it = re.finditer(self.startblock_tag_pattern, self.text)
        for mathc_obj in it:
            self.block_index_list.append(mathc_obj.span()[0])
        self.block_list = []

        for i in range(1, len(self.block_index_list)):
            self.block_list.append(self.text[self.block_index_list[i - 1]:self.block_index_list[i]])
        self.block_list.append(self.text[self.block_index_list[-1]:])
        self.blocks_groupby_type: Dict[ParserBase.BlockType, List[str]] = {t: [] for t in ParserBase.BlockType}
        for blk in self.block_list:
            self.blocks_groupby_type[self.get_block_type(blk)].append(blk)


def find_data_near_t(data_all_time, t, how_to_get_t=lambda data, i: data[i]['t'],
                     how_to_get_data=lambda data, i: data[i]['data']):
    """
    :param data_all_time: 按照时间升序排列，data[i]['t']应存在
    :param t:
    :return: 实际时间，对应的数据
    """
    delta_t = numpy.Inf
    for i in range(len(data_all_time)):
        new_delta_t = abs(how_to_get_t(data_all_time, i) - t)
        if new_delta_t <  delta_t:
            delta_t = new_delta_t
        else:
            i = max(i - 1, 0)
            break
    return how_to_get_t(data_all_time, i), how_to_get_data(data_all_time, i),i
