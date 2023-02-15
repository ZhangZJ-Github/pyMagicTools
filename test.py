# -*- coding: utf-8 -*-
# @Time    : 2023/2/15 15:47
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : test.py
# @Software: PyCharm
import re
fn = r"F:\MagicFiles\CherenkovAcc\test.txt"
txt = ""
with open (fn ,"r") as f:
    txt = f.read()

res = re.findall(r"\s+", txt)
pass