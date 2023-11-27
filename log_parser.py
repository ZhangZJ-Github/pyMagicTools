# -*- coding: utf-8 -*-
# @Time    : 2023/11/27 12:43
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : log_parser.py
# @Software: PyCharm
"""
用于解析Magic的log文件
"""
import typing

import _base


class LOG:
    def __init__(self, filename):
        self.filename = filename
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
