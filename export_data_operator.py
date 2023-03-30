# -*- coding: utf-8 -*-
# @Time    : 2023/3/28 10:39
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : export_data_operator.py
# @Software: PyCharm
"""
用于修改导出的par和grd
"""
import os.path
import re

import _base
import filenametool


def float2str(num, n_eff_digits: int = 7):
    """
    格式化为形如’0.123456E-12‘形式的字符串（以0开头）
    :return:
    """
    normal_str = ("%%.%dE" % (n_eff_digits - 1)) % (num)
    eff_part, pow = normal_str.split('E')
    if int(pow) + 1 < 0:
        pow_part = "E-%02d" % (-int(pow) - 1)
    else:
        pow_part = "E+%02d" % (int(pow) + 1)
    return ('%%.%df' % n_eff_digits) % (float(eff_part) / 10) + pow_part


class TimeModifierBase(_base.ParserBase):
    DEFAULT_N_CHAR = 400  # 需要修改的信息只位于前DEFAULT_N_CHAR个字符中

    def __init__(self, filename):
        super(TimeModifierBase, self).__init__(filename)

    def time_shift(self, delta_t, new_filename: str, blocktype: _base.ParserBase.BlockType):
        """
        将文件中的时间信息全部+delta_t，若修改后<0，则删除
        不会改变此对象
        即 新时刻 = 旧时刻 + delta_t
        :param delta_t:
        :return:
        """
        blocks = self.blocks_groupby_type[blocktype].copy()
        N = len(blocks)
        start_i = 0
        j = 0
        for i in range(N):
            block = blocks[i]
            str_to_change = block[:self.DEFAULT_N_CHAR]
            lines = str_to_change.splitlines(True)
            t = float(''.join(re.findall(_base.FrequentUsedPatter.float, lines[1])[0]))
            new_t = t + delta_t
            if new_t < 0:
                start_i = i + 1
                continue
            new_index = "%d" % (j + 1)
            N_space = len("%d" % (i + 1)) - len(new_index)
            new_index = " " * N_space + new_index
            lines[1] = re.sub(_base.FrequentUsedPatter.float, float2str(new_t, 7),
                              # "%.7e" % (new_t),  # lines[1],
                              re.sub("%d" % (i + 1), new_index, lines[1], 1),
                              1)
            if (''.join(lines) + block[self.DEFAULT_N_CHAR:] ).startswith(r''' $DMP$DMP$DMP$DMPSTARTBLOCK               1      
 FLUX               0.3433500E-11 54620E-10     9434  37'''):#j+1 == 546 and blocktype == _base.ParserBase.BlockType.FLUX:#lines[1] .startswith( ' FLUX               0.3433500E-11 54620E-10     9434  37'):
                raise RuntimeError
            j += 1
            blocks[i] = ''.join(lines) + block[self.DEFAULT_N_CHAR:]
        blocks = blocks[start_i:]
        header = self.text[:self.DEFAULT_N_CHAR].splitlines(True)[0]
        text = header + "".join(blocks)
        with open(new_filename, 'w') as f:
            f.write(text)


if __name__ == '__main__':
    filename = r"D:\MagicFiles\CherenkovAcc\cascade\min_case_for_gradient_test\test_export_and_import\test_export2_.grd"
    filename_no_ext = os.path.splitext(filename)[0]
    et = filenametool.ExtTool(filename_no_ext)
    grd = TimeModifierBase(filename)
    par = TimeModifierBase(et.get_name_with_ext(et.FileType.par))
    suffix = "_-trim"
    delta_t = -56e-12 - .7e-15  # - 4.8000000e-15 + 0.6300000E-14
    grd.time_shift(delta_t, filename_no_ext + suffix + ".grd", grd.BlockType.RANGE)
    par.time_shift(delta_t, filename_no_ext + suffix + ".par", grd.BlockType.FLUX)

    # grd.time_shift(-220e-12, filename_no_ext + "_-220ps.grd")
