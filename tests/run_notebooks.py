"""用当前 Python 作为内核执行全部示例，源码 notebook 不写入输出。"""
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def main():
    root = Path(__file__).resolve().parents[1]
    output_dir = root / 'assets' / 'executed_notebooks'
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(sorted((root / 'examples').glob('*.ipynb'))):
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        assert nb.cells[0].source.startswith('%load_ext autoreload\n%autoreload 3 --print --log')
        kernel = KernelManager(kernel_name='python3')
        kernel.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
        cwd = root if i % 2 == 0 else root / 'examples'
        print(f'执行 {path.name}（工作目录: {cwd.name}）', flush=True)
        client = NotebookClient(nb, km=kernel, timeout=120, resources={'metadata': {'path': str(cwd)}})
        try:
            client.execute()
        finally:
            # 外部提供 KernelManager 时 nbclient 不拥有其生命周期。
            if kernel.has_kernel:
                kernel.shutdown_kernel(now=True)
            kernel.cleanup_resources()
        nbformat.write(nb, output_dir / path.name)
        errors = [out for cell in nb.cells for out in cell.get('outputs', []) if out.output_type == 'error']
        assert not errors, errors
        print(f'通过 {path.name}', flush=True)
    print(f'已执行全部示例；副本位于 {output_dir}', flush=True)


if __name__ == '__main__':
    main()
