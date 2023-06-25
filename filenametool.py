# -*- coding: utf-8 -*-
# @Time    : 2023/3/28 11:50
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : filenametool.py
# @Software: PyCharm
import re
import enum
class ExtTool:
    """
    Magic常见结果文件的后缀
    """

    class FileType(enum.Enum):
        par = ".par"
        fld = ".fld"
        grd = ".grd"
        m2d = ".m2d"
        toc = '.toc'
        geom_png = '.geom.png'  # 手动截图的建议后缀

    def __init__(self, filename_no_ext):
        self.filename_no_ext = filename_no_ext

    def get_name_with_ext(self, ext: enum.Enum):
        return self.filename_no_ext + ext.value


def validateTitle(title):
    """
    将文本替换为合法的文件夹名字
    :param title:
    :return:
    """
    rstr = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
    new_title = re.sub(rstr, "_", title)  # 替换为下划线
    return new_title
