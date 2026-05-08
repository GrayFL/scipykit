from scipykit.utils import *
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import dataset, DataLoader, Subset
import random
from tqdm import tqdm
import sys
from datetime import datetime

from typing import Literal
from abc import ABCMeta, abstractmethod


class ModelAbstract(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def train(self, *args, **kwds):
        pass

    @abstractmethod
    def test(self, *args, **kwds):
        pass


class BaseNetworker():
    '''
    这是一个基础的网络，包含了以下功能：
    - 加载、分割数据集
    - 设置batchsize、epoches等参数
    - 训练与测试
    - 定义学习率曲线
    - 记录训练时的log
    - 验证单条数据

    默认使用GPU
    '''

    def __init__(
            self,
            model_cls,
            batch_size: int = 512,
            num_epochs: int = 100,
            dataloader_train: DataLoader = None,
            dataloader_test: DataLoader = None,
            model_dtype=torch.float32,
            loss_fn: nn.Module = None,
            optimizer=None,
            optimizer_cls=torch.optim.AdamW,
            optimizer_cfg: dict = None,
            verbose_interval_train: int = 4,
            verbose_interval_test: int = 2,
            checkpoint_interval: int = None,
            device: Union[str, torch.device] = None,
            **kwds
        ) -> None:
        '''
        Parameters
        ---
        model_cls:
            - 定义模型的类
        optimizer_cls:
            - 优化器的类
        verbose_interval_train:
            - 训练阶段的log次数
        verbose_interval_test:
            - 测试阶段的log次数
        '''
        torch.cuda.empty_cache()
        self.model_cls = model_cls
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.model_dtype = model_dtype
        self._loss_fn = loss_fn
        self.optimizer = optimizer or None
        self.optimizer_cls = optimizer_cls
        self.optimizer_cfg = optimizer_cfg or {}
        self.verbose_interval_train = verbose_interval_train
        self.verbose_interval_test = verbose_interval_test
        self.checkpoint_interval = checkpoint_interval
        self.log: dict[str, list] = {}
        self.progress = 0
        self.current_epoch = 0

        if isinstance(device, str):
            device = torch.device(device)
        self.DEVICE = device or torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'
            )

        self.reset_net()
        # self.model.to(self.DEVICE)
        # self.optimizer = self.optimizer_cls(self.model.parameters())

    @property
    def loss_fn(self):
        if None is not self._loss_fn:
            return self._loss_fn
        elif hasattr(self.model, 'loss_fn'):
            return self.model.loss_fn
        else:
            raise ValueError('No loss function defined')

    @loss_fn.setter
    def loss_fn(self, loss_fn):
        self._loss_fn = loss_fn

    def load_dataset(
            self,
            dataset: dataset.Dataset,
            which: Literal['train', 'test'] = 'test',
            batch_size: int = None,
            **kwds
        ):
        self.batch_size = batch_size or self.batch_size
        if 'train' == which:
            self.dataloader_train = DataLoader(
                dataset, batch_size=self.batch_size, **kwds
                )
        elif 'test' == which:
            self.dataloader_test = DataLoader(
                dataset, batch_size=self.batch_size, **kwds
                )

    def reset_net(self):
        self.model: ModelAbstract = self.model_cls()
        torch.cuda.empty_cache()
        self.model.to(self.DEVICE)
        if None is self.optimizer:
            print('optimizer is None, create new optimizer')
            self.optimizer = self.optimizer_cls(
                params=self.model.parameters(),
                **self.optimizer_cfg,
                )

    def train(self):
        print('train', '=' * 18)
        arr_loss = []
        amount_batches = len(self.dataloader_train)
        size_data = len(self.dataloader_train.dataset)
        self.model.train()
        for ibatch, (n_Xs, n_Ys) in enumerate(self.dataloader_train):
            n_Xs: torch.Tensor = n_Xs.to(self.DEVICE)
            n_Ys: torch.Tensor = n_Ys.to(self.DEVICE)
            n_Ys_pred = self.model(n_Xs)
            train_loss: torch.Tensor = self.loss_fn(n_Ys_pred, n_Ys)
            num_train_loss = train_loss.item()
            arr_loss.append(num_train_loss)

            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()

            if ibatch % (amount_batches // self.verbose_interval_train
                            or 1) == 0:
                print(
                    f'{ibatch*self.batch_size:<5}/{size_data:>5} loss: {num_train_loss}'
                    )
        self.log['train_loss'].append(np.mean(arr_loss))

    def test(self):
        print('test', '=' * 18)
        arr_loss = []
        amount_batches = len(self.dataloader_test)
        size_data = len(self.dataloader_test.dataset)
        self.model.eval()
        with torch.no_grad():
            for ibatch, (n_Xs, n_Ys) in enumerate(self.dataloader_test):
                n_Xs: torch.Tensor = n_Xs.to(self.DEVICE)
                n_Ys: torch.Tensor = n_Ys.to(self.DEVICE)
                n_Ys_pred = self.model(n_Xs)
                test_loss: torch.Tensor = self.loss_fn(n_Ys_pred, n_Ys)
                num_test_loss = test_loss.item()
                arr_loss.append(num_test_loss)
                if ibatch % (amount_batches // self.verbose_interval_test
                                or 1) == 0:
                    print(
                        f'{ibatch*self.batch_size:<5}/{size_data:>5} loss: {num_test_loss}'
                        )
        self.log['test_loss'].append(np.mean(arr_loss))
        print(f'Avg. loss: {np.mean(arr_loss)}')

    def train_model(
        self,
        num_epochs: int,
        lr_fn=lambda _: 0.001,
        loss_fn: nn.Module = None,
        verbose_interval_train: int = None,
        verbose_interval_test: int = None,
        checkpoint_interval: int = None,
        checkpoint_filepath: str = None,
        omit_lr_fn: bool = False,
        ):
        BaseNetworker.setup_seed(0)
        self.num_epochs = num_epochs or self.num_epochs
        self.loss_fn = loss_fn or self.loss_fn
        self.verbose_interval_train = \
            verbose_interval_train or self.verbose_interval_train
        self.verbose_interval_test = \
            verbose_interval_test or self.verbose_interval_test
        self.checkpoint_interval = \
            checkpoint_interval or self.checkpoint_interval

        self.log['train_loss'] = []
        self.log['test_loss'] = []
        for t in tqdm(
                range(num_epochs),
                total=num_epochs,
                file=sys.stdout,
            ):
            print(f"\nEpoch {t+1}\n-------------------------------")
            self.current_epoch = t
            if omit_lr_fn:
                lr = self.optimizer.param_groups[0]['lr']
            else:
                lr = lr_fn(t / num_epochs)
                self.optimizer.param_groups[0]['lr'] = lr
            self.progress = t / num_epochs
            print(f'lr: {lr:.6f}')
            self.train()
            self.test()
            if None is not checkpoint_filepath:
                Path(str(checkpoint_filepath)).mkdir(
                    parents=True, exist_ok=True
                    )
                if (None is not self.checkpoint_interval
                   ) and (t % self.checkpoint_interval == 0):
                    BaseNetworker.save_model(self, checkpoint_filepath)
                elif t == num_epochs - 1:
                    BaseNetworker.save_model(self, checkpoint_filepath)
        print("Done!")

    @staticmethod
    def setup_seed(seed):
        import os
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现

    def pred(
            self,
            input: torch.Tensor,
            dtype=torch.float32
        ) -> torch.Tensor:
        input = input.to(self.DEVICE)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(input)
            if isinstance(preds, tuple):
                preds = [pred.detach().cpu() for pred in preds]
            elif isinstance(preds, torch.Tensor):
                preds = preds.detach().cpu()
        return preds

    @staticmethod
    def save_model(
            networker: "BaseNetworker",
            filepath: Union[str, Path],
            addition=''
        ):
        """
        Notes
        ---
        保存模型的参数，不保存模型的结构。
        建议只输入工作文件夹而不指定文件名
        """
        model_name = networker.model_cls.__name__
        cur_epoch = networker.current_epoch + 1
        cur_time = datetime.now().strftime('%Y%m%d%H%M%S')
        num_epochs = networker.num_epochs
        latest_loss = networker.log['test_loss'][-1]
        filename = f'{model_name}_{cur_time}_[eph={cur_epoch}of{num_epochs}]_[loss={latest_loss:<.3f}]_{addition}.pth'
        if not isinstance(filepath, Path):
            filepath = Path(filepath)
        if filepath.suffix == '':
            # 说明是文件夹
            filepath.mkdir(parents=True, exist_ok=True)
            filepath = filepath / filename
        else:
            # 说明是文件名
            # filepath = filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            assert filepath.suffix == '.pth', '文件后缀必须为.pth'
        # pkl_save(networker.model, filepath)
        torch.save(networker.model.state_dict(), str(filepath))

    @staticmethod
    def load_model(
            networker: "BaseNetworker",
            filepath: Union[str, Path],
            optimizer: None
        ):
        # model = pkl_load(filepath)
        model: ModelAbstract = networker.model_cls()
        model.load_state_dict(
            torch.load(filepath, map_location=networker.DEVICE)
            )
        networker.model = model
        if None is not optimizer:
            networker.optimizer = optimizer
        else:
            networker.optimizer = networker.optimizer_cls(
                params=networker.model.parameters(),
                **networker.optimizer_cfg,
                )
