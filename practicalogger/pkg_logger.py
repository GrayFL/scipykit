from scipykit.utils import pkl_load, pkl_save
from collections import deque
from datetime import datetime
from pathlib import Path
from colorama import Back, Fore, Style


class LogFrame():
    """
    单次log的结构定义
    记录完成后记得使用 `pack` 方法输出为字典
    """

    def __init__(self, data: dict = None, **kwds):
        # if None is data:
        #     data = kwds
        # else:
        #     assert isinstance(data, dict), f"LogFrame init with wrong type: {type(data)} {data}"
        #     data.update(kwds)

        # self.data = data
        self.data: dict[str] = {}
        self.add(data, **kwds)

    def add(self, data: dict = None, **kwds):
        if None is data:
            self.data.update(kwds)
        else:
            assert isinstance(data, dict), f"LogFrame init with wrong type: {type(data)} {data}"
            data.update(kwds)
            self.data.update(data)

    def pack(self):
        time = datetime.now().strftime(r"%Y/%m/%d - %H:%M:%S")
        package = {'pack_time': time, **self.data}
        self.data.clear()
        return package

    def __getitem__(self, key):
        return self.data.__getitem__(key)

    def __repr__(self):
        return self.data.__repr__()


class PkgLogger(deque):
    """
    双端队列
    """

    def __init__(self, save_path="./logs"):
        super().__init__(self)
        save_path = Path(save_path)
        self.save_path = save_path

    def append(self, frame):
        if not isinstance(frame, LogFrame):
            frame = LogFrame(frame)
        return super().append(frame.pack())

    def save(self, file_path: str = None):
        file_path = file_path or self.save_path
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        file_path.mkdir(exist_ok=True)
        try:
            pkl_save(self, file_path / "~pctclog.pkl")
            (file_path / "~pctclog.pkl").replace(file_path / "pctclog.pkl")
            print(
                Fore.LIGHTBLUE_EX
                + f"successfully save to {file_path / 'pctclog.pkl'}"
                )
        except Exception as e:
            print(e)

    @staticmethod
    def load(file_path: str = './logs'):
        # file_path = file_path or self.save_path
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        if (file_path / "pctclog.pkl").exists():
            obj = pkl_load(file_path / "pctclog.pkl")
            return obj
        else:
            obj = PkgLogger(save_path=file_path)
            obj.save()
            return obj
