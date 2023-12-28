# -*- coding: utf-8 -*-
# @Time    : 2023/11/27 13:40
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : spectrum_diff.py
# @Software: PyCharm
import os
import re

import matplotlib

matplotlib.use('tkagg')

matplotlib.rcParams['font.family'] = 'SimHei'

import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
import grd_parser

plt.ion()

dir = r"E:\GeneratorAccelerator\Genac\optmz\Genac10G50keV\粗网格\单独处理\单周期 腔个数"
filenames = os.listdir(dir)
filenames.sort(key=lambda filename: int(filename[:-len('.grd')]))

for filename in filenames:
    if filename.endswith('.grd'):
        grd = grd_parser.GRD(os.path.join(dir,
                                          filename))

        obs_titles = []

        for key in grd.obs.keys():
            if re.findall(r'FIELD E1 @SWS[0-9]+\.[0-9]+', key) and re.findall(r'FFT-#[0-9]+\.2', key):
                obs_titles.append(key)
                # logger.info(key)
        print(obs_titles)
        # obs_titles = [' FIELD E1 @SWS1.1.SLOT.OBSPT,FFT-#14.2', ' FIELD E1 @SWS1.2.SLOT.OBSPT,FFT-#17.2',
        #               ' FIELD E1 @SWS1.3.SLOT.OBSPT,FFT-#20.2', ' FIELD E1 @SWS2.1.SLOT.OBSPT,FFT-#23.2',
        #               ' FIELD E1 @SWS2.2.SLOT.OBSPT,FFT-#26.2', ' FIELD E1 @SWS2.3.SLOT.OBSPT,FFT-#29.2',
        #               ' FIELD E1 @SWS3.1.SLOT.OBSPT,FFT-#32.2', ' FIELD E1 @SWS3.2.SLOT.OBSPT,FFT-#35.2',
        #               ' FIELD E1 @SWS3.3.SLOT.OBSPT,FFT-#38.2', ' FIELD E1 @SWS3.4.SLOT.OBSPT,FFT-#41.2']

        plt.figure()
        for key in obs_titles:
            plt.plot(*grd.obs[key]['data'].values.T, label=key)
        plt.legend()
        plt.xlim([0, 20])
        plt.title(filename)
