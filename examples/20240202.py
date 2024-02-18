import matplotlib
from matplotlib import pyplot as plt

plt.ion()
import grd_parser

from geom_parser import GEOM

# geom = GEOM(r"E:\BigFiles\GENAC\GENACX50kV\手动\GenacX50kV - beam - thin beam - 去直流.grd")
geom = GEOM(r"E:\BigFiles\GENAC\GENACX50kV\optimizing\GenacX50kV_tmplt_20240210_051249_02.grd")

grd = geom.grd
ranges = grd.ranges[r' FIELD EZ @LINE_PARTICLE_MOVING$,FFT #3.1'][-30:]

fig, axs = plt.subplots(2, 1, sharex=True)
geom.plot(axs[0])
for rg in ranges:
    axs[1].plot(*rg['data'].values.T, label=rg['t'])
axs[1].legend()

from examples.average import range_time_average

avg_P_particle = range_time_average(grd.ranges[' PARTICLE ALL @ALONG AXIS #7.1'][-60:])
plt.figure()
plt.plot(*avg_P_particle)
