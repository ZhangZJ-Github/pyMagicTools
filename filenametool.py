# -*- coding: utf-8 -*-
# @Time    : 2023/3/28 11:50
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : filenametool.py
# @Software: PyCharm
import re
import enum
import os
import shutil
class ExtTool:
    """
    Magic常见结果文件的后缀
    """

    class FileType(enum.Enum):
        # TODO: 此处似乎不必用枚举类
        par = ".par"
        fld = ".fld"
        grd = ".grd"
        m2d = ".m2d"
        toc = '.toc'
        log = '.log'
        geom_png = '.geom.png'
        png = '.png'

    def __init__(self, filename_no_ext):
        self.filename_no_ext = filename_no_ext

    def get_name_with_ext(self, ext: enum.Enum):
        return self.filename_no_ext + ext.value
    @staticmethod
    def from_filename(filename:str):
        """
        从有后缀的文件名构造此对象
        """
        filename_no_ext = os.path.splitext(
            filename
        )[0]
        return ExtTool(filename_no_ext)


def validateTitle(title):
    """
    将文本替换为合法的文件夹名字
    :param title:
    :return:
    """
    rstr = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
    new_title = re.sub(rstr, "_", title)  # 替换为下划线
    return new_title
def get_default_result_dir_name_from_filename(filename: str):
    """
    返回某个MAGIC文件对应的结果目录
    例如，A:/B/C/D.toc对应的默认结果目录为A:/B/C/D/
    """
    filename_no_ext = os.path.splitext(
        filename
    )[0]
    et = ExtTool(filename_no_ext)
    return "%s/.out/%s" % (os.path.split(et.filename_no_ext))

def copy_m2d_to_result_dir(res_dir_name: str, et: ExtTool, ):
    copy_to_result_dir(et.get_name_with_ext(ExtTool.FileType.m2d), res_dir_name)


def copy_to_result_dir(sourcce_path, destination_dir, ):
    destination_path = os.path.join(destination_dir, os.path.split(sourcce_path)[1])
    shutil.copyfile(sourcce_path, destination_path)
