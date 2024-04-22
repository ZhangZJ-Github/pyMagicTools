import matplotlib
from matplotlib import pyplot as plt

plt.ion()
import grd_parser

from geom_parser import GEOM

# geom = GEOM(r"E:\BigFiles\GENAC\GENACX50kV\手动\GenacX50kV - beam - thin beam - 去直流.grd")
geom = GEOM(r"E:\BigFiles\GENAC\GENACX50kV\手动\GenacX50kV_tmplt_20240210_051249_02\GenacX50kV_tmplt_20240210_051249_02 rang I1.m2d")

grd = geom.grd


aaaaaa
In_JDA_titles_prefix = ' FIELD_INTEGRAL J.DA @OSYS$AREA #'
In_JDA_titles = []
for each in grd.ranges:
    if each .startswith(In_JDA_titles_prefix):
        In_JDA_titles.append(each)
In_JDA_titles = In_JDA_titles[1:]

fig,axs = plt.subplots(2,1,sharex = True)
# plt.figure()
# ax= plt.gca()
# ax2 = plt.twinx()
# axs = [ax,ax2]
idx = -1
from _logging import  logger
geom.plot(axs[0])
axs[0].set_ylim(30e-3,None)
JDAs = 0
for i, title in enumerate( In_JDA_titles):
    range_= grd.ranges[title][idx]
    JDAs+=( range_['data'][1])
    axs[1].plot(*range_['data'].values.T,label =  "$I_{%s}$"%(i))
    # axs[1].set_ylim(0,150)
logger.info(      range_['t'])

axs[1].plot(  range_['data'][0], JDAs,label = 'sum')
plt.legend()











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
