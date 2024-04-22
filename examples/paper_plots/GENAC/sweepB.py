# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/19 11:16
@Auth ： Zi-Jing Zhang (张子靖)
@File ：sweepB.py
@IDE ：PyCharm
"""
import pandas
import matplotlib.pyplot as plt

plt.ion()
df = pandas.read_excel(r'E:\BigFiles\GENAC\GENACX50kV\optimizing\数据处理.xlsx', sheet_name='磁场影响')
df = df[pandas.isna(df['comment2'])# &( df["%Bref%"]<=1.5)
]

plt.figure(figsize=(4, 3), constrained_layout=True)
plt.plot(df["%Bref%"], df["power efficiency score"], '.-')
plt.xlabel("magnetic induction intensity / T")
plt.ylabel("power efficiency")
plt.ylim(0, 0.35)
# plt.grid()
plt.savefig("power_eff_Bz.png", dpi=400)

from brokenaxes import brokenaxes
fig = plt.figure(figsize=(4, 2), constrained_layout=True
           # tight_layout = True
           )
bax = brokenaxes(xlims=[(-0.1, 1.4),(3.8,4.2)], ylims=[(0,0.35)],
                 despine=False,
                 d = 0.01,
                 fig= fig
                 )
bax.plot(df["%Bref%"], df["power efficiency score"], '.-')
# bax.set_xlabel("magnetic induction intensity / T")
# bax.set_ylabel("power efficiency")
bax.grid()
plt.savefig("power_eff_Bz.png", dpi=400)
