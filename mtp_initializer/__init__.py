"""科研绘图便捷入口；首次导入自动应用历史 rcParams 默认预设。"""
from pathlib import Path
import numpy as np
import matplotlib as mtp
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.projections.polar import PolarAxes
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.gridspec import GridSpec
from .annotator import (
    text_better,
    find_annotate_position,
    find_annotate_positions,
    select_artists,
    clear_mono_metric_cache
    )
from .colors import get_cmap, mtp_get_cmap, with_alpha, palette, PALETTES
from .style import plot_style, set_plot_style, configure_inline, STYLES, DEFAULT_RC_PARAMS
from .figure import mfigure, subplots, savefig, clipfig, canvas_to_array
from .display import disp, display, display_markdown, clr, IPyImage, inspect_notebook_name
from .geometry import (
    xywh2xyxy,
    xyxy2xywh,
    get_pixels_per_data_unit,
    scale_data2pt,
    scale_pt2data
    )
from .images import show_image, draw_box, draw_text, draw_pts
from .elements import mark_on_axes, label_axes, panel_label, plot_band, add_colorbar

set_plot_style()

__all__ = [
    'np',
    'Path',
    'mtp',
    'plt',
    'Axes',
    'PolarAxes',
    'Axes3D',
    'GridSpec',
    'text_better',
    'find_annotate_position',
    'find_annotate_positions',
    'select_artists',
    'clear_mono_metric_cache',
    'get_cmap',
    'mtp_get_cmap',
    'with_alpha',
    'palette',
    'PALETTES',
    'plot_style',
    'set_plot_style',
    'configure_inline',
    'STYLES',
    'DEFAULT_RC_PARAMS',
    'mfigure',
    'subplots',
    'savefig',
    'clipfig',
    'canvas_to_array',
    'disp',
    'display',
    'display_markdown',
    'clr',
    'IPyImage',
    'inspect_notebook_name',
    'xywh2xyxy',
    'xyxy2xywh',
    'get_pixels_per_data_unit',
    'scale_data2pt',
    'scale_pt2data',
    'show_image',
    'draw_box',
    'draw_text',
    'draw_pts',
    'mark_on_axes',
    'label_axes',
    'panel_label',
    'plot_band',
    'add_colorbar',
    ]
