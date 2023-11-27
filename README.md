# MAGIC后处理工具
## 快速入门
不同后缀的文件用对应的parser解析即可，例如，解析par，应采用par_parser.py。

total_parser.py可以对同一.m2d文件生成的几种结果文件进行整合处理。

下面的例子展示了如何解析一个.grd文件：

```python
import grd_parser
grd = grd_parser.GRD(r'path/to/data.grd')

# How to use a GRD object
print(grd.obs.keys()) # 查看此.grd文件中包含的observe数据标题

print(grd.ranges[list(grd.ranges.keys())[0]][0]['data'] # 查看此.grd文件中包含的第0块range数据

```

更合适的使用方法是在PyCharm的控制台执行以上代码，随后可以在变量列表中查看grd对象的结构。

![388d67f0aaddbda7172e2ab6196ed14d.png](.md_attachments/388d67f0aaddbda7172e2ab6196ed14d.png)

## 兼容性

暂时只支持Magic 2005 Single，double版本及Magic 2017版运行结果中会用1.0+12这种形式来表示浮点数1.0E+12，暂时无法解析。


## 效果展示

利用total_parser.py输出组图到指定文件夹：

![b67036a28b4cc03a04cce9a76479ac12.png](.md_attachments/b67036a28b4cc03a04cce9a76479ac12.png)

![9e33fd7468c0e5fcbaf011fcaaa8338b.png](.md_attachments/9e33fd7468c0e5fcbaf011fcaaa8338b.png)

![157bbc844ecd317494579af2d3d45acb.png](.md_attachments/157bbc844ecd317494579af2d3d45acb.png "157bbc844ecd317494579af2d3d45acb.png")


paper_plot/trivial.py输出精修图：

![d41d8953c378defb8069edf251030904.png](.md_attachments/d41d8953c378defb8069edf251030904.png)

![678f635c7c15cbeb7722e0baea3e4581.png](.md_attachments/678f635c7c15cbeb7722e0baea3e4581.png)
更多数据展示方式：
![a1cb5069d056212f136e1e048782e834.png](.md_attachments/a1cb5069d056212f136e1e048782e834.png)

![238b54c8d91709e4b6038700eaf6b8d4.png](.md_attachments/238b54c8d91709e4b6038700eaf6b8d4.png)
![a072f7196d44eb93e270a42f12203c37.png](.md_attachments/a072f7196d44eb93e270a42f12203c37.png)

（上面的示例中，几何结构通过贴图实现，须事先保存几何结构的截图）

## 绘制几何结构

```python
from geom_parser import GEOM
import matplotlib.pyplot as plt
filename = r"D:\MagicFiles\CherenkovAcc\cascade\min_case_for_gradient_test\test_diffraction-23.grd"
# filename = r"E:\GeneratorAccelerator\Genac\optmz\Genac10G50keV\粗网格\单独处理\Genac10G50keV2.grd"
geom = GEOM(filename)
plt.figure()
geom.plot(plt.gca())
```

![9d6730f8c7ae9904e71d4806d55abb68.png](.md_attachments/9d6730f8c7ae9904e71d4806d55abb68.png)
![03843dd168382254da1ce501cce9a4d1.png](.md_attachments/03843dd168382254da1ce501cce9a4d1.png)
