"""基础文件 IO 与目录工具。"""
from pathlib import Path
from typing import Literal, Union
import json
import pickle
import shutil

def sizeof(var: object, backend: Literal['pympler', 'sys'] = 'pympler'):
    if 'pympler' == backend:
        from pympler.asizeof import asizeof
        size = asizeof(var)
    elif 'sys' == backend:
        from sys import getsizeof
        size = getsizeof(var)
    if (size // 1024) == 0:
        return f'{size} B'
    elif (size // 1024**2) == 0:
        return f'{size/1024:.2f} KB'
    elif (size // 1024**3) == 0:
        return f'{size/1024**2:.2f} MB'


def pkl_save(obj, file_path: str):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def pkl_load(file_path: str):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def json_save(obj, file_path: str, **kwds):
    with open(file_path, 'w') as f:
        json.dump(obj, f, ensure_ascii=False, **kwds)


def json_load(file_path: str):
    with open(file_path, 'r') as f:
        return json.load(f)


def copy_to(
        src_path: Union[str, Path],
        dst_path: Union[str, Path],
        verbose=False
    ):
    """
    Copy file or directory to **dst_path**.
    """
    if isinstance(src_path, str):
        src_path = Path(src_path)
    if isinstance(dst_path, str):
        dst_path = Path(dst_path)
    assert src_path.exists(), f'{src_path} not exists'
    if dst_path.exists():
        if dst_path.is_file():
            raise FileExistsError(f'{dst_path} is a file')
    if src_path.is_file():
        if verbose:
            print(f'copy file {src_path} to {dst_path}')
        dst = dst_path / src_path.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst)
        return
    elif src_path.is_dir():
        if verbose:
            print(f'copy directory {src_path} to {dst_path}')
        dst = dst_path / src_path.name
        # dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_path, dst, dirs_exist_ok=True)
        return


def generate_tree(
        path: Union[str, Path],
        max_depth=3,
        max_dirs: int = None,
        max_items=3,
        ignores: list[str] = []
    ):
    if isinstance(path, str):
        path = Path(path)

    def _generate_tree(path: Path, n=0, is_last=False):
        for ignore in ignores:
            if path.match(ignore):
                return ''
        if path.is_file():
            if is_last:
                _str = '╎   ' * (n-1) + '└───' + '┴── ' + path.name + '\n'
            else:
                _str = '╎   ' * (n) + '├── ' + path.name + '\n'
            return _str

        elif path.is_dir():
            if (None is not max_depth) and n >= max_depth:
                return '╎   ' * (n-1) + '└ ' + '...\n'
            _str = '╎   '*n + '├ ' + path.name + '/' + '\n'
            length = len(list(path.iterdir()))
            count_dirs = 0
            count_items = 0
            for _i, cp in enumerate(path.iterdir()):
                if cp.is_dir():
                    count_dirs += 1
                    if ((None is not max_dirs) and count_dirs > max_dirs):
                        _str += '╎   ' * (n) + '└───' + '┴── ' + '...\n'
                        break
                else:
                    count_items += 1
                    if ((None is not max_items)
                            and count_items > max_items):
                        _str += '╎   ' * (n) + '└───' + '┴── ' + '...\n'
                        break
                is_last = _i == length - 1
                _str += _generate_tree(cp, n + 1, is_last)
            return _str

    tree_str = _generate_tree(path, 0)
    return tree_str

