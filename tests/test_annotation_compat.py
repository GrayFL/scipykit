"""当前 Matplotlib API 的针对性回归；原有算法测试保持原样。"""
import matplotlib.pyplot as plt
import numpy as np
from scipykit.mtp_initializer import plot_style, text_better
from scipykit.mtp_initializer.annotator import text_artist_uses_complex_layout


def test_normal_multiline_layout_uses_exact_fallback():
    with plot_style():
        fig, ax = plt.subplots()
        try:
            probe = ax.text(.5, .5, '第一行\n第二行', linespacing='normal')
            assert text_artist_uses_complex_layout(probe, {})
            probe.remove()
            common = dict(avoid_artists=False,
                          placement_kwargs={'preferred_direction': (0, 0)})
            fast = text_better(fig, ax, .5, .5, '第一行\n第二行',
                               placement_mode='mono_fast', **common)
            exact = text_better(fig, ax, .5, .5, '第一行\n第二行',
                                placement_mode='exact', **common)
            np.testing.assert_allclose(fast.get_position(), exact.get_position())
        finally:
            plt.close(fig)


def test_monospace_single_line_still_supports_fast_mode():
    with plot_style():
        fig, ax = plt.subplots()
        try:
            artist = text_better(fig, ax, .5, .5, '标注 A', placement_mode='mono_fast',
                                 avoid_artists=False)
            fig.canvas.draw()
            assert np.isfinite(artist.get_window_extent().extents).all()
        finally:
            plt.close(fig)
