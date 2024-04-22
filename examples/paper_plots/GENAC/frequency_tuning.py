# -*- coding: utf-8 -*-
"""
@Time ： 2024/2/22 22:47
@Auth ： Zi-Jing Zhang (张子靖)
@File ：frequency_tuning.py
@IDE ：PyCharm
"""
import matplotlib.pyplot as plt
import numpy
import pandas
import grd_parser
import filenametool
import json

delta_f = pandas.read_excel(
    r"E:\BigFiles\GENAC\GENACX50kV\optimizing\数据处理.xlsx",
    sheet_name='改腔径调频（已筛选）',
    # encoding='GBK'
)

out_spectra_title = r' FIELD_INTEGRAL E.DL @COUPLER.PORT,FFT-#13.2'
spectra_data = {}

for i in delta_f.index:
    grd = grd_parser.GRD(
        filenametool.ExtTool.from_filename(delta_f['m2d_path'][i]).get_name_with_ext(filenametool.ExtTool.FileType.grd))
    spectra_data[i] = (
        delta_f['%ddrout%'][i], json.loads(delta_f['freq peaks'][i]), delta_f['power efficiency score'][i],
        grd.obs[out_spectra_title]['data'])

fs = numpy.array([(spectra_data[i][0], spectra_data[i][1][1][0] / 2) for i in spectra_data])
sorted_index = numpy.argsort(fs[:, 0])

# fs = fs[sorted_index]
power_effs = numpy.array([(spectra_data[i][0], spectra_data[i][2]) for i in spectra_data])  # [sorted_index]

import matplotlib.colors as mcolors

color_table = mcolors.TABLEAU_COLORS

plt.figure(figsize=(4,2), constrained_layout=True)
c1, c2 ,c3= list(color_table.keys())[:3]
ax1 = plt.gca()
ax1.plot(*fs[sorted_index].T, 'o-', c=c1
         )
ax1.yaxis.label.set_color(c1)
ax1.tick_params(axis='y', colors=c1)
ax1.spines['left'].set_color(c1)


# ax1.set_xlim(-0.8, 0.6)
# ax1.set_ylim(8.7, 10.2)
ax1.set_xlabel('$\Delta r$ / mm')
ax1.set_ylabel('frequency / GHz')
ax1.set_xlim(-0.3,0.3)

ax2 = ax1.twinx()
ax2.plot(*power_effs[sorted_index].T,  # linewidth = 2
         'v-', c=color_table[list(color_table.keys())[1]]
         )
ax2.set_ylabel("power efficiency")

ax2.yaxis.label.set_color(c2)
ax2.tick_params(axis='y', colors=c2)
ax2.spines['right'].set_color(c2)
ax2.spines['left'].set_color(c1)
ax2.set_ylim(0, 0.35)
# ax3= ax1.twiny()
height_spectra = 0.05
for i in sorted_index:
    spectrum = spectra_data[i][3]
    delta_f = 0.12
    _filter = (spectrum[0] < fs[i, 1] + delta_f) & (spectrum[0] > fs[i, 1] - delta_f)
    ax1.plot(((spectrum[1] / max(spectrum[1]) - 1) * height_spectra + fs[i, 0])[_filter] - 0.015,
             spectrum[0][_filter], linewidth=0.5,c =c1)

# ax3.set_xlim(0, 1.2)
# ax3.set_ylim(8.3, 10.3)
# ax3.set_xlabel( "intensity / arb. unit")
# ax3:plt.Axes
# ax3.yaxis.set_axis_off()

plt.savefig("tuning_curve.png",dpi = 400)
