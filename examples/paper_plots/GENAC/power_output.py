# -*- coding: utf-8 -*-
"""
@Time ： 2024/2/25 21:58
@Auth ： Zi-Jing Zhang (张子靖)
@File ：power_output.py
@IDE ：PyCharm
"""
import geom_parser
import matplotlib.pyplot as plt
import pandas


# import simulation.optimize.hpm.hpm
def _get_mean(df: pandas.DataFrame, DeltaT):
    """
    获取近周期的时间序列数据df在时间间隔DeltaT内的均值
    :param df: 第0列为时间，第1列为值
    :param DeltaT:
    :return:
    """
    colname_period = 'period'
    df[colname_period] = df[0] // (DeltaT)
    return df.groupby(colname_period).mean()  # .iloc[:-2] # 倒数第二个周期的平均功率。倒数第一个周期可能不全，结果波动很大，故不取。


geom = geom_parser.GEOM(
    r"E:\BigFiles\GENAC\GENACX50kV\手动\GenacX50kV_tmplt_20240210_051249_02\GenacX50kV_tmplt_20240210_051249_02 rang I1.m2d")
grd = geom.grd
powerdata = grd.obs[' FIELD_POWER S.DA @COUPLER.PORT,FFT-#5.1']['data']
VdataFD= grd.obs[' FIELD_INTEGRAL E.DL @COUPLER.PORT,FFT-#13.2']['data']

plt.figure(figsize=(2.5, 2),                            constrained_layout=True)
plt.plot(powerdata[0], -powerdata[1] / 1e6, label='transient',linewidth = 0.005)
plt.xlim(0,max(powerdata[0]))
plt.ylim(0,None)
f = 9.46e9
mean_power = _get_mean(powerdata, 5 * 1 / f)
mean_power[1] /= -1e6
plt.plot(*mean_power.values.T, label='period average')
# plt.legend()
plt.gca().xaxis.set_major_formatter(lambda z, pos: '%d' % (z / 1e-9))


plt.xlabel('time / ns')
plt.ylabel('power output / MW')
plt.grid()

plt.savefig('power_out.png', dpi=400)

plt.figure(figsize=(2, 2),   constrained_layout=True)
plt.plot(VdataFD[0],VdataFD[1] / VdataFD[1].max(),)
plt.xlabel('frequency / GHz')
plt.ylabel('intensity / arb. unit')
plt.xlim(0,40)
plt.ylim(0, 1.2)
plt.grid()
plt.savefig('V_out_FFT.png',dpi = 400)
