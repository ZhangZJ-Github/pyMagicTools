# -*- coding: utf-8 -*-
# @Time    : 2023/11/27 12:43
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : log_parser.py
# @Software: PyCharm
"""
用于解析Magic的log文件
"""
import re
import time

import pandas
import pandas as pd

from _logging import logger


class LOG:
    def __init__(self, filename):
        t = time.time()
        self.filename = filename
        self.text = ""
        with open(self.filename, 'r') as f:
            self.text = f.read()
        t1 = time.time()
        logger.info("将文件%s读入内存（未处理）耗时：%.2f" % (filename, t1 - t))
        self.geom_structure = self.parse_geom_generator_result(self.text)

    @staticmethod
    def parse_geom_generator_result(text: str) -> pandas.DataFrame:
        materialtype_table_start = r'===> Structure Generator 2d <==='
        it = re.finditer(materialtype_table_start, text)
        for match in it:
            sta = match.span()[1] + 1
        materialtype_table_end = '> ================================================\n          ... Testing for Boundary type OUTGOING'
        it = re.finditer(materialtype_table_end, text)
        for match in it:
            end = match.span()[0]
        list = text[sta:end].split('\n')
        for i in range(len(list)):
            list[i] = list[i].split()
        columns = list[0]
        del list[0]
        df = pd.DataFrame(list, columns=columns)
        df = df.drop(df.shape[0] - 1)
        return df


if __name__ == '__main__':
    filename = r"E:\GeneratorAccelerator\Genac\optmz\Genac10G50keV\粗网格\单独处理\Genac10G50keV2.log"
    log = LOG(filename)
