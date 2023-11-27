# -*- coding: utf-8 -*-
# @Time    : 2023/11/27 12:43
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : log_parser.py
# @Software: PyCharm
"""
用于解析Magic的log文件
"""
import pandas as pd
import typing
import _base
import enum
import re
from _logging import logger
import time
import typing
from io import StringIO
from typing import List

import matplotlib
import shapely.geometry



class LOG:
    def __init__(self, filename):
        t = time.time()
        self.filename = filename
        self.text = ""
        with open(self.filename, 'r') as f:
            self.text = f.read()
        t1 = time.time()
        logger.info("将文件%s读入内存（未处理）耗时：%.2f" % (filename, t1 - t))
        self.geom_structure = self.parse_object_materialtype()
        # TODO: 参考其他parser的__init__

    def parse_object_materialtype(self) -> typing.OrderedDict[str, _base.MaterialType]:
        """

        :return: 有序字典（ordered_dict），形式为
        {
            object1: type_of_object1,
            object2: type_of_object2,
            ...
        }
        如
        {
            'CATHODE': CONDUCTOR,
            'SWS1.slot1': VOID
        }

        """
        # TODO:
        self.materialtype_table_start = r'===> Structure Generator 2d <==='
        it = re.finditer(self.materialtype_table_start, self.text)
        for match in it:
            sta=match.span()[1]+1
        self.materialtype_table_end = '> ================================================\n          ... Testing for Boundary type OUTGOING'
        it = re.finditer(self.materialtype_table_end, self.text)
        for match in it:
            end = match.span()[0]
        list = self.text[sta:end].split('\n')
        for i in range(len(list)):
            list[i] = list[i].split()
        columns = list[0]
        del list [0]
        df = pd.DataFrame(list,columns = columns)
        df = df.drop(df.shape[0]-1)
        return df







if __name__ == '__main__':

    filename = r"E:\11-24\2\template_20231127_072536_54.log"
    log = LOG(filename)