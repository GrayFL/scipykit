import numpy as np
import pandas as pd
from scipykit.sci_initializer import json_save, json_load, pkl_save, pkl_load, pd_read_markdown_table
from scipykit.task.file import json_save as task_json_save


def test_legacy_file_aggregation_and_roundtrip(tmp_path):
    assert task_json_save is json_save
    obj = {'中文': [1, 2], '参数': .3}
    json_save(obj, tmp_path / 'config.json', indent=2)
    assert json_load(tmp_path / 'config.json') == obj
    array = np.arange(6).reshape(2, 3)
    pkl_save(array, tmp_path / 'array.pkl')
    np.testing.assert_array_equal(pkl_load(tmp_path / 'array.pkl'), array)


def test_markdown_table_preserves_empty_cells_and_removes_styling():
    frame = pd_read_markdown_table('| 方法 | 备注 |\n| --- | --- |\n| **A** | |\n| `B` | [链接](https://example.com) |')
    pd.testing.assert_frame_equal(frame, pd.DataFrame({'方法': ['A', 'B'], '备注': ['', '链接']}))
