"""
关于字号和磅的对应：

| 中文字号 | 英文字号（磅）/pt |
|----------|-------------------|
| 初号     | 42                |
| 小初     | 36                |
| 一号     | 26                |
| 小一     | 24                |
| 二号     | 22                |
| 小二     | 18                |
| 三号     | 16                |
| 小三     | 15                |
| 四号     | 14                |
| 小四     | 12                |
| 五号     | 10.5              |
| 小五     | 9                 |
| 六号     | 7.5               |
| 小六     | 6.5               |
| 七号     | 5.5               |
| 八号     | 5                 |
"""

# import os
from matplotlib.cm import get_cmap as mtp_get_cmap
import matplotlib as mtp
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.projections.polar import PolarAxes
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.gridspec import GridSpec
from IPython.display import display as disp, clear_output as clr
# plt.rcParams['font.family'] = 'Times New Roman,Simsun'
plt.rcParams['font.family'] = 'Sarasa Mono SC'
plt.rcParams['font.size'] = 10.5  # 10.5pt 五号字；9pt 小五号字
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['figure.subplot.left'] = 0.11  #0.05
plt.rcParams['figure.subplot.right'] = 0.89  #0.98
plt.rcParams['figure.subplot.bottom'] = 0.15  #0.1
plt.rcParams['figure.subplot.top'] = 0.91  #0.93
plt.rcParams['figure.facecolor'] = (1, 1, 1, 0)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.facecolor'] = (1, 1, 1, 0)
plt.rcParams['axes.titlesize'] = 'medium'
plt.rcParams['legend.fontsize'] = 'medium'
plt.rcParams['interactive'] = 'False'
# plt.rcParams['color_cycle'] = ['b','g','r','c','m','y','k']
import numpy as np

# import matplotlib.backends.backend_agg
import matplotlib_inline.backend_inline

matplotlib_inline.backend_inline.set_matplotlib_formats(
    *['png', 'jpeg'],  # *['png', 'jpeg', 'svg']
    # *args 表示激活哪些格式的输出（也就是vscode里的更改演示文稿，默认只打开png），'jpg'、'jpeg'等效
    bbox_inches=None,  # 默认是jupyter会设置为tight，会在mfigure外产生空白，所以要设置为 None
    format='png',  # 默认的display格式
    pil_kwargs={
        'quality': 80,  # 图片质量，0-100，越大质量越好。只能对JPEG格式生效
        'optimize': True,  # 对GIF的优化
        'compress_level': 9,  # PNG的压缩级别，0-9，越大压缩越好。但似乎不起作用
        },
    )

plt.plot()
plt.ioff()
plt.close()

from typing import Literal

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_cmap(name: Literal['Accent', 'Blues', 'BrBG', 'BuGn', 'BuPu', 'CMRmap', 'Dark2', 'GnBu', 'Greens', 'Greys', 'OrRd', 'Oranges', 'PRGn', 'Paired', 'Pastel1', 'Pastel2', 'PiYG', 'PuBu', 'PuBuGn', 'PuOr', 'PuRd', 'Purples', 'RdBu', 'RdGy', 'RdPu', 'RdYlBu', 'RdYlGn', 'Reds', 'Set1', 'Set2', 'Set3', 'Spectral', 'Wistia', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd', 'afmhot', 'autumn', 'binary', 'bone', 'brg', 'bwr', 'cividis', 'cool', 'coolwarm', 'copper', 'cubehelix', 'flag', 'gist_earth', 'gist_gray', 'gist_heat', 'gist_ncar', 'gist_stern', 'gist_yarg', 'gnuplot', 'gnuplot2', 'gray', 'hot', 'hsv', 'inferno', 'jet', 'magma', 'nipy_spectral', 'ocean', 'pink', 'plasma', 'prism', 'rainbow', 'seismic', 'spring', 'summer', 'tab10', 'tab20', 'tab20b', 'tab20c', 'terrain', 'twilight', 'twilight_shifted', 'viridis', 'winter'], **kwds): #yapf: disable
    """
    Parameters
    ---
    name:
        - 预设热力图 `'viridis', 'plasma', 'inferno', 'magma', 'cividis'`
        - 同色系亮度递减 `'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'`
        - 亮度变化更剧烈 `'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper'`
        - 中间为白色 `'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'`
        - 环形色相 `'twilight', 'twilight_shifted', 'hsv'`
        - 量化色盘 `'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c'`
        - 其他 `'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral', 'gist_ncar'`

    Others
    ---
    'Accent', 'Blues', 'BrBG', 'BuGn', 'BuPu', 'CMRmap', 'Dark2', 'GnBu', 'Greens', 'Greys', 'OrRd', 'Oranges', 'PRGn', 'Paired', 'Pastel1', 'Pastel2', 'PiYG', 'PuBu', 'PuBuGn', 'PuOr', 'PuRd', 'Purples', 'RdBu', 'RdGy', 'RdPu', 'RdYlBu', 'RdYlGn', 'Reds', 'Set1', 'Set2', 'Set3', 'Spectral', 'Wistia', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd', 'afmhot', 'autumn', 'binary', 'bone', 'brg', 'bwr', 'cividis', 'cool', 'coolwarm', 'copper', 'cubehelix', 'flag', 'gist_earth', 'gist_gray', 'gist_heat', 'gist_ncar', 'gist_stern', 'gist_yarg', 'gnuplot', 'gnuplot2', 'gray', 'hot', 'hsv', 'inferno', 'jet', 'magma', 'nipy_spectral', 'ocean', 'pink', 'plasma', 'prism', 'rainbow', 'seismic', 'spring', 'summer', 'tab10', 'tab20', 'tab20b', 'tab20c', 'terrain', 'twilight', 'twilight_shifted', 'viridis', 'winter'
    """
    return mtp_get_cmap(name, **kwds)


def mfigure(
        figsize=(6.66, 3),
        dpi=300,
        frame=True,
        face_color=(1, 1, 1, 0),
        alpha=0,
        **kdic
    ):
    fig = plt.figure(figsize=figsize, dpi=dpi, **kdic)
    fig.set_facecolor((1, 1, 1, alpha))
    bed: Axes = fig.add_axes([0, 0, 1, 1])
    bed.set_frame_on(frame)
    bed.set_xticks([])
    bed.set_yticks([])
    bed.set_facecolor(face_color)
    return fig


def show_image(
        img,
        scale=1.0,
        size_factor=120,
        is_output=False,
        is_to_RGB=False,
        **kwds
    ):
    '''
    简单地，一次性地，在jupyter中绘制一张图片

    Parameters
    ----------
    scale : folat
        默认情况下采用dpi=120=size_factor，即1920x1080 -> 16x9; 通过scale可以等比例控制画布大小
    '''
    h, w, *_ = img.shape
    if is_to_RGB:
        import cv2
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dpi = size_factor * scale
    fig = mfigure(
        figsize=(w / size_factor, h / size_factor), dpi=dpi, frame=False
        )
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(img, **kwds)
    disp(fig)
    if is_output:
        fig.canvas.draw()
        frame = fig.canvas.buffer_rgba()
        plt.close(fig)
        return frame
    plt.close(fig)


def savefig(
        fig: plt.Figure,
        file_path: str,
        is_frame_on=False,
        alpha=1.0,
        **kwds
    ):
    """
    注意 该函数仅仅适用于使用mfigure创建的fig
    """
    fig.get_axes()[0].set_frame_on(is_frame_on)
    fig.set_facecolor((1, 1, 1, alpha))
    fig.savefig(file_path, **kwds)


def clipfig(fig, ext='jpg', **kwds):
    savefig(fig, f'./__temp__.{ext}', **kwds)
    import subprocess
    args = ['powershell', f'Get-Item ./__temp__.{ext} | Set-Clipboard']
    subprocess.Popen(args=args)


def xywh2xyxy(xywh: list):
    return [xywh[0], xywh[1], xywh[2] + xywh[0], xywh[3] + xywh[1]]


def xyxy2xywh(xywh: list):
    return [xywh[0], xywh[1], xywh[2] - xywh[0], xywh[3] - xywh[1]]


def get_pixels_per_data_unit(ax: Axes, invert=False):
    """Calculate pixels per data unit in x and y axes."""
    # Transform (0,0) and (1,0)/(0,1) to pixel coordinates
    input_dims = ax.transData.input_dims
    tmp = input_dims - 1
    tar = ([0] * tmp + [1]) * 2
    if not invert:
        func_transtorm = ax.transData.transform
    else:
        func_transtorm = ax.transData.inverted().transform
    ori_0 = func_transtorm([0] * input_dims)
    tar_axis = []
    for i in range(input_dims):
        tar_axis.append(func_transtorm(tar[i:i + input_dims])[i])

    d_axis = (np.array(tar_axis)[::-1] - ori_0)[::-1]
    return d_axis


def scale_data2pt(ax: Axes, data: float):
    """
    计算数据坐标系中的长度在pt中的长度
    """
    d_axis = get_pixels_per_data_unit(ax)
    dpi = ax.figure.dpi
    size_in_pt_xyz = data * d_axis * (72/dpi)
    return size_in_pt_xyz


def scale_pt2data(ax: Axes, pt: float):
    """
    计算pt在数据坐标系中的长度
    """
    d_axis = get_pixels_per_data_unit(ax, invert=True)
    dpi = ax.figure.dpi
    size_in_data_xyz = pt * (dpi/72) * d_axis
    return size_in_data_xyz
