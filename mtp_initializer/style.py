"""历史 rcParams 默认预设，以及可选的科研绘图样式。"""
import matplotlib as mpl

# 与重构前入口中的 rcParams 一致；首次导入绘图包时自动应用。
DEFAULT_RC_PARAMS = {
    'font.family': ['Inconsolata', 'Sarasa Mono SC'],
    'font.size': 10.5,
    'mathtext.fontset': 'stix',
    'figure.subplot.left': 0.11,
    'figure.subplot.right': 0.89,
    'figure.subplot.bottom': 0.15,
    'figure.subplot.top': 0.91,
    'figure.facecolor': (1, 1, 1, 0),
    'figure.dpi': 300,
    'axes.facecolor': (1, 1, 1, 0),
    'axes.titlesize': 'medium',
    'legend.fontsize': 'medium',
    'interactive': False,
}

STYLES = {
    'default': DEFAULT_RC_PARAMS,
    'paper': {
        'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'savefig.bbox': None,
    },
    'notebook': {'font.size': 12, 'axes.grid': True, 'grid.alpha': 0.2},
    'presentation': {'font.size': 16, 'lines.linewidth': 2.5},
}


def _params(preset, overrides):
    if preset not in STYLES:
        raise ValueError(f'未知样式 {preset!r}，可选: {tuple(STYLES)}')
    return {**DEFAULT_RC_PARAMS, **STYLES[preset], **overrides}


def plot_style(preset='default', **overrides):
    """with plot_style('paper', **{'font.size': 9}): ...，退出后恢复。"""
    return mpl.rc_context(_params(preset, overrides))


def set_plot_style(preset='default', **overrides):
    """切换会话样式或重新应用默认预设；默认使用时无需调用。"""
    mpl.rcParams.update(_params(preset, overrides))


def configure_inline(formats=('png',), **kwargs):
    """在 notebook 中显式设置输出格式；不切换 backend。"""
    from matplotlib_inline.backend_inline import set_matplotlib_formats
    set_matplotlib_formats(*formats, bbox_inches=None, **kwargs)
