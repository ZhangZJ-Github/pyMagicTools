# -*- coding: utf-8 -*-
# @Time    : 2023/4/2 14:19
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : show_diff_between_exported_and_imported.py
# @Software: PyCharm
"""
显示导出数据与导入数据的区别
"""
import matplotlib

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.interpolate import interp1d
import numpy
import grd_parser

exported_grd = grd_parser.GRD(
    r"D:\MagicFiles\CherenkovAcc\cascade\min_case_for_gradient_test\test_export_and_import\test_export2.grd")
imported_grd = grd_parser.GRD(
    r"D:\MagicFiles\CherenkovAcc\cascade\min_case_for_gradient_test\test_export_and_import\test_import2.grd")

ex_obs_name = r' FIELD E1 @EXPORT_LINE-#53.1'
im_obs_name = r' FIELD E1 @IMPORT_LINE-#50.1'
exported_data = exported_grd.obs[ex_obs_name]['data']
imported_data = imported_grd.obs[im_obs_name]['data']
fig, ax = plt.subplots(  # 3, 1, sharex=True,
    constrained_layout=True)
axs = [ax]
axs[0].plot(exported_data[0] - 40e-12, exported_data[1], label="exported")
axs[0].plot(*imported_grd.obs[im_obs_name]['data'].values.T, label="imported")
axs[0].set_xlabel('t / s')

axs[0].set_ylabel('E / (V/m)')
axs[0].legend()
exported_40ps_later = exported_data[exported_data[0] >= 40e-12]
# axs[1].plot(exported_40ps_later[0] - 40e-12,
#             exported_40ps_later[1].values / imported_data.values[:len(exported_40ps_later), 1])
# axs[2].plot(exported_40ps_later[0] - 40e-12, exported_40ps_later[1].values)
# axs[2].plot(exported_40ps_later[0] - 40e-12, imported_data.values[:len(exported_40ps_later), 1])
fig2, axs2 = plt.subplots(2, 1, sharex=True, constrained_layout=True)
freqs_exported = fftpack.fftfreq(len(exported_40ps_later), exported_data[0][1] - exported_data[0][0])
fft_exported = numpy.abs(fftpack.fft(exported_40ps_later[1].values))
N = len(exported_40ps_later) // 2
axs2[0].plot(freqs_exported[:N], fft_exported[:N], label='exported')
freqs_imported = fftpack.fftfreq(len(imported_data), imported_data[0][1] - imported_data[0][0])
fft_imported = numpy.abs(fftpack.fft(imported_data[1].values))
axs2[0].plot(freqs_imported[:N],
             fft_imported[:N], label='imported')
axs2[0].legend()
f = interp1d(freqs_imported, fft_imported, kind='linear')
axs2[1].plot(freqs_exported[:N], fft_exported[:N] / f(freqs_exported[:N]), label='exported / imported')
_x=  numpy.linspace(0, 1e12, 20)
axs2[1].plot(_x, numpy.ones(_x.shape)*2.0,'--')

axs2[1].set_xlim(0, 1e12)
axs2[1].set_ylim(0, 4)

plt.legend()
plt.show()
