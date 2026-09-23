import importlib
import os
import subprocess
import sys
from pathlib import Path
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from scipykit.mtp_initializer import (
    mfigure, savefig, canvas_to_array, get_cmap, palette, with_alpha,
    plot_style, mark_on_axes, plot_band, scale_data2pt, scale_pt2data,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def test_star_imports_and_shared_implementations():
    plotting, cv = {}, {}
    exec('from scipykit.mtp_initializer import *', plotting)
    exec('from scipykit.task.cv import *', cv)
    for name in ('mfigure', 'disp', 'savefig', 'canvas_to_array', 'draw_box',
                 'draw_text', 'draw_pts', 'mark_on_axes', 'with_alpha', 'text_better'):
        assert plotting[name] is cv[name]
    assert not {'CameraTools3D', 'Frustum', 'camera', 'frustum'} & cv.keys()
    assert 'os' not in plotting


@pytest.mark.parametrize('entry', ['scipykit.mtp_initializer', 'scipykit.task.cv'])
def test_import_applies_legacy_rcparams_without_creating_figures(entry):
    code = '''
import os
import matplotlib.pyplot as plt
plt.ion()
fig = plt.figure()
old = dict(plt.rcParams)
os.environ['NOTEBOOK_ASSETS_ROOT'] = 'my-assets'
exec('from ' + __import__('sys').argv[1] + ' import *')
assert plt.rcParams['font.family'] == ['Inconsolata', 'Sarasa Mono SC']
assert plt.rcParams['font.size'] == 10.5
assert plt.rcParams['mathtext.fontset'] == 'stix'
assert plt.rcParams['figure.dpi'] == 300
assert [plt.rcParams['figure.subplot.' + side] for side in ('left', 'right', 'bottom', 'top')] == [.11, .89, .15, .91]
assert plt.rcParams['figure.facecolor'] == (1, 1, 1, 0)
assert plt.rcParams['axes.facecolor'] == (1, 1, 1, 0)
assert plt.rcParams['axes.titlesize'] == 'medium'
assert plt.rcParams['legend.fontsize'] == 'medium'
assert not plt.isinteractive()
assert plt.rcParams['axes.spines.top'] == old['axes.spines.top']
assert plt.rcParams['axes.spines.right'] == old['axes.spines.right']
assert plt.get_fignums() == [fig.number]
assert os.environ['NOTEBOOK_ASSETS_ROOT'] == 'my-assets'
if __import__('sys').argv[1] == 'scipykit.mtp_initializer':
    assert 'cv2' not in __import__('sys').modules
# Python 缓存已导入的模块，重复 import * 不覆盖用户后续的配置。
plt.rcParams['font.size'] = 13
exec('from ' + __import__('sys').argv[1] + ' import *')
assert plt.rcParams['font.size'] == 13
'''
    subprocess.run([sys.executable, '-c', code, entry], check=True, env={**os.environ, 'MPLBACKEND': 'Agg'})


def test_user_workflow_reuses_path_and_preserves_figure(tmp_path, monkeypatch):
    module = importlib.import_module('scipykit.mtp_initializer.display')
    shown = []
    monkeypatch.setattr(module, 'display', lambda *a, **kw: shown.append((a, kw)))
    fig = mfigure((6, 3), dpi=150, alpha=1)
    ax = fig.add_subplot()
    ax.plot([0, 1], [0, 1])
    old = fig.get_facecolor(), [a.get_frame_on() for a in fig.axes], ax.get_facecolor()
    path = module.disp(fig, 'name', assets_root=tmp_path, notebook_name='demo')
    assert path == tmp_path / 'demo/name.png'
    assert module.IMAGE_MIME in shown[0][0][0]
    assert 'text/html' in shown[0][0][0]
    assert savefig(fig, dpi=300, alpha=0) == path
    with Image.open(path) as img:
        assert img.size == (1800, 900)
        assert img.getpixel((0, 0))[3] == 0
    assert old == (fig.get_facecolor(), [a.get_frame_on() for a in fig.axes], ax.get_facecolor())


def test_save_plain_figure_file_object_and_restore_on_failure(tmp_path, monkeypatch):
    fig, ax = plt.subplots()
    buffer = BytesIO()
    assert savefig(fig, buffer, format='png', alpha=0) is buffer
    assert buffer.getvalue().startswith(b'\x89PNG')
    explicit = BytesIO()
    savefig(fig, explicit, format='png', transparent=True)
    explicit.seek(0)
    with Image.open(explicit) as img:
        assert img.getpixel((0, 0))[3] == 0
    fig2 = mfigure()
    monkeypatch.setattr(fig2, 'savefig', lambda *a, **k: (_ for _ in ()).throw(OSError('failure')))
    with pytest.raises(OSError):
        savefig(fig2, tmp_path / 'broken.png')
    assert fig2.axes[0].get_frame_on()


def test_default_path_and_canvas_before_display(tmp_path, monkeypatch):
    monkeypatch.setenv('NOTEBOOK_ASSETS_ROOT', str(tmp_path))
    fig = mfigure((2, 1), dpi=100)
    arr = canvas_to_array(fig, 'RGBA')
    assert arr.shape == (100, 200, 4)
    assert arr.flags.owndata
    assert savefig(fig) == tmp_path / 'figure.png'
    with pytest.raises(ValueError):
        canvas_to_array(fig, 'BGR')


def test_dataframe_external_output_and_standard_fallback(tmp_path, monkeypatch):
    module = importlib.import_module('scipykit.mtp_initializer.display')
    shown = []
    monkeypatch.setattr(module, 'display', lambda data, **kw: shown.append(data))
    df = pd.DataFrame({'中文': range(100)})
    module.disp(df)
    assert set(shown[-1]) == {'text/plain'}
    paths = module.disp(df, 'table', df_formats=('csv', 'xlsx'), assets_root=tmp_path)
    pd.testing.assert_frame_equal(pd.read_csv(paths['csv'], index_col=0), df)
    assert '<td>99</td>' in paths['html'].read_text()
    assert 'text/html' not in shown[-1]
    assert len(shown[-1][module.DF_MIME]['data']) == 2
    auto_paths = module.disp(df, 'auto', assets_root=tmp_path)
    assert {'csv', 'parquet'} & auto_paths.keys()
    module.disp(df, 'no-html', df_formats='csv', df_link_html=False, assets_root=tmp_path)
    assert module.DF_MIME in shown[-1]
    with pytest.raises(ValueError):
        module.disp(df, '../escape', assets_root=tmp_path)


@pytest.mark.parametrize('relative_root', [None, 'custom-assets'])
def test_private_mime_paths_are_absolute_and_survive_chdir(tmp_path, monkeypatch, relative_root):
    module = importlib.import_module('scipykit.mtp_initializer.display')
    shown = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('NOTEBOOK_ASSETS_ROOT', 'assets')
    monkeypatch.setattr(module, 'display', lambda data, **kw: shown.append(data))
    options = dict(assets_root=relative_root, notebook_name='demo')
    fig = mfigure((2, 1), dpi=50)
    image_path = module.disp(fig, 'image', **options)
    assert Path(shown[-1][module.IMAGE_MIME]['image']) == image_path
    frame = pd.DataFrame({'中文': [1, 2], 'value': [.1, .2]})
    files = module.disp(frame, 'table', df_formats=('parquet', 'csv', 'xlsx'), **options)
    descriptor = shown[-1][module.DF_MIME]
    paths = [image_path, *files.values(), *map(Path, descriptor['data'].values()), Path(descriptor['html'])]
    assert all(path.is_absolute() and path.exists() for path in paths)
    pd.testing.assert_frame_equal(pd.read_parquet(files['parquet']), frame)
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert all(path.exists() for path in paths)
    assert savefig(fig) == image_path


def test_auto_dataframe_format_uses_installed_pyarrow(tmp_path, monkeypatch):
    module = importlib.import_module('scipykit.mtp_initializer.display')
    monkeypatch.setattr(module, 'display', lambda *a, **kw: None)
    frame = pd.DataFrame({'label': ['中文', 'A'], 'score': [.9, .8]})
    paths = module.disp(frame, 'auto', assets_root=tmp_path)
    assert set(paths) == {'parquet', 'html'}
    pd.testing.assert_frame_equal(pd.read_parquet(paths['parquet']), frame)


def test_colors_elements_style_and_units():
    assert get_cmap('viridis', lut=4).N == 4
    assert len(palette('okabe_ito', 12)) == 12
    assert len(palette('plasma', 3)) == 3
    assert with_alpha('red', .3) == (1, 0, 0, .3)
    old = plt.rcParams['font.size']
    with plot_style('paper', **{'font.size': 8}):
        assert plt.rcParams['font.size'] == 8
    assert plt.rcParams['font.size'] == old
    fig, ax = plt.subplots()
    boxes = mark_on_axes(ax, [[0, 0, 1, 1], [1, 1, 2, 2]], edgecolor=[1., 0., 0.])
    assert all(b.get_edgecolor() == (1, 0, 0, 1) for b in boxes)
    line, band = plot_band(ax, [0, 1], [1, 2], std=.2)
    assert line.axes is ax and band.axes is ax
    ax.set(xlim=(10, -2), ylim=(-20, 40))
    np.testing.assert_allclose(scale_pt2data(ax, scale_data2pt(ax, .5)), [.5, .5])
