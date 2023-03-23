# -*- coding: utf-8 -*-
# @Time    : 2023/2/15 15:47
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : test.py
# @Software: PyCharm
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

plt.plot([1, 2], [3, 4])
with open(r"D:\MagicFiles\CherenkovAcc\cascade\min_case_for_gradient_test\test_export_and_import\export_Z45grd",
          'r') as f:
    lines = f.readlines()
used_lines = [lines[0]] + lines[31790935:-1]
# used_str = "".join(used_lines)
with open(
        r"D:\MagicFiles\CherenkovAcc\cascade\min_case_for_gradient_test\test_export_and_import\export_Z45_lasttime_.grd",
        'w') as f:
    f.writelines(used_lines)
