import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import dataset, DataLoader, Subset

import numpy as np
import pandas as pd

from typing import Literal, Union, Callable

from pathlib import Path
import sys
from tqdm import tqdm
# from tqdm.notebook import tqdm
from datetime import datetime
import time

from functools import partial
from abc import ABCMeta, abstractmethod

from scipykit.task.file import *


class ModelAbstract(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def loss_fn(self, n_Ys_pred, n_Ys):
        pass

    @abstractmethod
    def backward_fn(self, loss):
        pass


class Networker():

    def __init__(
            self,
            model_builder: Callable,
            optimizer_builder: Callable = torch.optim.AdamW,
            runtime_fn: Callable = lambda s, *x: x,
            lr_fn: Callable = lambda t: 1e-3,
            loss_fn: Callable = None,
            num_epochs: int = None,
            batch_accumulation: int = 1,
            pipeline: list[dict[str, int]] = [ \
                ('train', 1), ('test', 1)
                ],
            device: Literal['cpu', 'cuda'] = 'cpu',
            **kwds
        ):
        self.model_builder = model_builder
        self.optimizer_builder: torch.optim.Optimizer = optimizer_builder
        self.device = device
        self.runtime_fn = runtime_fn
        self.loss_fn = loss_fn
        self.batch_accumulation = batch_accumulation
        self.model = None
        self.num_epochs = num_epochs
        self.schedule_pattern = pipeline
        self.current_epoch = 0
        self._mode: Literal['train', 'test'] = 'train'
        self._optim_references = [{'optimizer': None, 'lr_fns': [None]}]
        self.lr_fn = lr_fn
        self.log: list[dict[str]] = []
        self.log_filepath = None
        pass

    def build_model(self, **kwds):
        self.model: ModelAbstract = self.model_builder(**kwds)
        self.model.to(self.device)

    def build_optimizer(self, **kwds):
        self.optimizer: torch.optim.Optimizer = self.optimizer_builder(
            params=self.model.parameters(), lr=self.lr_fn(0), **kwds
            )

    def build_networker(self, **kwds):
        self.build_model()
        self.build_optimizer()

    @property
    def runtime_fn(self):
        return self._runtime_fn

    @runtime_fn.setter
    def runtime_fn(self, runtime_fn: Callable):
        self._runtime_fn = partial(runtime_fn, self)

    @property
    def loss_fn(self):
        '''
        loss_fn 大多数时候来自模型的内部定义，不预先包含 networker 数据，所以适合只与模型有关的那种 loss 计算
        '''
        if None is not self._loss_fn:
            return self._loss_fn
        else:
            assert hasattr(self.model, 'loss_fn'), 'model has no loss_fn'
            return self.model.loss_fn

    @loss_fn.setter
    def loss_fn(self, loss_fn: Callable):
        if None is loss_fn:
            self._loss_fn = None
            print(
                '[Warning] loss_fn is not set, use model.loss_fn instead'
                )
        else:
            self._loss_fn = loss_fn

    def backward_fn(
            self,
            loss: Union[
                torch.Tensor,
                list[torch.Tensor],
                dict[str, torch.Tensor],
                ],
            batch_accumulation: int = 1,
            **kwds
        ):
        '''
        由于 backward_fn 是内置函数，默认带有 self ，因此一些与模型本身相关的参数，比如运行轮数、loss 记录等可以通过重写 backward_fn 来实现
        '''
        if isinstance(loss, torch.Tensor):
            (loss / batch_accumulation).backward()
        elif isinstance(loss, list):
            for _i, l in enumerate(loss):
                (l / batch_accumulation).backward(
                    retain_graph=True if _i < len(loss) - 1 else False
                    )
        elif isinstance(loss, dict):
            for _i, l in enumerate(loss.values()):
                (l / batch_accumulation).backward(
                    retain_graph=True if _i < len(loss) - 1 else False
                    )

    # def step_lr(self):
    #     for param_group in self.optimizer.param_groups:
    #         param_group['lr'] = self.lr_fn(self.progress)

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode: Literal['train', 'test']):
        self._mode = mode
        if mode == 'train':
            self.model.train()
        elif mode == 'test':
            self.model.eval()

    @property
    def progress(self):
        return self.current_epoch / self.num_epochs

    @property
    def optim_references(self):
        '''
        多优化器且多参数组，每个参数组对应一个学习率函数

        形如：
        ```python
        [
            {
                'optimizer': torch.optim.Optimizer,
                'lr_fns': [
                    lambda t: 1e-3,
                    lambda t: 1e-4,
                    lambda t: 1e-5, 
                    ],
                ...
                } 
            ]
        ```
        '''
        return self._optim_references

    @optim_references.setter
    def optim_references(
        self,
        optim_references: list[
            dict[Union[str, int], Union[torch.optim.Optimizer, Callable]],
            ]
        ):
        assert isinstance(optim_references, list), '[Error] optim_references 是列表'
        for optim_reference in optim_references:
            assert optim_reference['optimizer'], '[Error] optim_reference 需包含 optimizer 键'
            assert isinstance(
                optim_reference['optimizer'], torch.optim.Optimizer), '[Error] optimizer 需为 torch.optim.Optimizer'
            assert optim_reference['lr_fns'], '[Error] optim_reference 需包含 lr_fns 键'
            assert isinstance(optim_reference['lr_fns'], list), '[Error] lr_fns 需为列表'
            for lr_fn in optim_reference['lr_fns']:
                assert callable(lr_fn), '[Error] lr_fn 需为可调用对象'
        self._optim_references = optim_references

    @property
    def optimizer(self):
        return self.optim_references[0]['optimizer']

    @optimizer.setter
    def optimizer(self, optimizer: torch.optim.Optimizer):
        self.optim_references[0]['optimizer'] = optimizer

    @property
    def lr_fn(self):
        return self.optim_references[0]['lr_fns'][0]

    @lr_fn.setter
    def lr_fn(self, lr_fn: Callable):
        self.optim_references[0]['lr_fns'][0] = lr_fn

    @property
    def lr_fns(self):
        return self.optim_references[0]['lr_fns']

    def init_lr(self):
        for optim_reference in self.optim_references:
            optimizer = optim_reference['optimizer']
            for param_group, lr_fn in zip(
                optim_reference['optimizer'].param_groups, optim_reference['lr_fns']
                ):
                param_group['lr'] = lr_fn(0)
            optimizer.zero_grad()

    def step_lr(self, is_step_optimizer=True):
        for optim_reference in self.optim_references:
            optimizer = optim_reference['optimizer']
            for param_group, lr_fn in zip(
                optimizer.param_groups, optim_reference['lr_fns']
                ):
                param_group['lr'] = lr_fn(self.progress)
            optimizer.step()
            optimizer.zero_grad()

    def train_epoch(
            self,
            dataloader_train: DataLoader,
            batch_accumulation=1,
            visualization_interval=10,
            visualization_amount=None,
            is_drop_last=False,
            **kwds
        ):
        self.mode = 'train'
        amount_batches = len(dataloader_train)
        amount_samples = len(dataloader_train.dataset)
        count_samples = 0
        verbose_infos = {}
        if None is not visualization_amount:
            visualization_interval = (
                int(
                    np.ceil(
                        np.ceil(amount_batches / batch_accumulation)
                        / visualization_amount
                        )
                    ) or 1
                )
        self.init_lr()

        fmt = "{desc} {bar} {n_fmt:>3}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        pbar = tqdm(
            total=(amount_batches//batch_accumulation) +
            (amount_batches%batch_accumulation//1),
            # 防止最后一个 batch 不被更新
            mininterval=0.0,
            miniters=1,
            # unit='batches',
            file=sys.stdout,
            bar_format=fmt,
            )
        pbar.set_description_str(
            f'Epoch {self.current_epoch+1:02} {self.mode.capitalize():5} ',
            refresh=False
            )

        for ibatch, (n_Xs, n_Ys) in enumerate(dataloader_train):
            count_samples += len(n_Xs)
            is_need_step_optimizer = ( #
                (((ibatch+1) % batch_accumulation) == 0)
                or ((ibatch == (amount_batches - 1)) and not is_drop_last)
                )

            n_Xs, n_Ys = self.runtime_fn(n_Xs, n_Ys)
            n_Ys_pred = self.model(n_Xs)
            loss = self.loss_fn(n_Ys_pred, n_Ys)
            self.backward_fn(loss, batch_accumulation=batch_accumulation)
            if is_need_step_optimizer:
                self.step_lr()
                pbar.update()

            verbose_infos.update(self.get_lr_infos())
            verbose_infos.update(getattr(self.model, 'verbose_infos', {}))
            if isinstance(loss, dict):
                loss = loss.get('loss', next(iter(loss.values())))
            pbar.set_postfix_str(
                f'[loss: {loss.item():.6f} '
                f'samples: {count_samples}/{amount_samples}] '
                f'[{self.format_verbose_infos(verbose_infos)}] '
                f'[{sum([sem.get_value() for sem in dataloader_train.dataset.iterator.dic_sem_slot_loaded.values()]) if hasattr(dataloader_train.dataset.iterator, "dic_sem_slot_loaded") else ""}]',
                refresh=False
                )
            pbar.refresh()
            if (((ibatch//batch_accumulation) + 1) % visualization_interval
                    == 0) and is_need_step_optimizer:
                self.log_infos(
                    self.current_epoch,
                    ibatch,
                    loss.item(),
                    **verbose_infos
                    )
                print()
        pbar.close()

    def test_epoch(
            self,
            dataloader_test: DataLoader,
            visualization_interval=10,
            visualization_amount=None,
            is_drop_last=False,
            **kwds
        ):
        self.mode = 'test'
        amount_batches = len(dataloader_test)
        amount_samples = len(dataloader_test.dataset)
        count_samples = 0
        avg_loss = 0
        if None is not visualization_amount:
            visualization_interval = (
                int(np.ceil(amount_batches / visualization_amount)) or 1
                )
        verbose_infos = {}

        fmt = "{desc} {bar} {n_fmt:>3}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        pbar = tqdm(
            total=amount_batches,
            mininterval=0.0,
            miniters=1,
            # unit='batches',
            file=sys.stdout,
            bar_format=fmt,
            )
        pbar.set_description_str(
            f'Epoch {self.current_epoch+1:02} {self.mode.capitalize():5} ',
            refresh=False
            )
        with torch.no_grad():
            for ibatch, (n_Xs, n_Ys) in enumerate(dataloader_test):
                if (ibatch == (amount_batches - 1)) and is_drop_last:
                    break
                count_samples += len(n_Xs)

                n_Xs, n_Ys = self.runtime_fn(n_Xs, n_Ys)
                n_Ys_pred = self.model(n_Xs)
                loss = self.loss_fn(n_Ys_pred, n_Ys)
                if isinstance(loss, dict):
                    loss = loss.get('loss', next(iter(loss)))
                avg_loss = (avg_loss*ibatch + loss.item()) / (ibatch+1)

                verbose_infos.update(self.get_lr_infos())
                verbose_infos.update(
                    getattr(self.model, 'verbose_infos', {})
                    )
                pbar.set_postfix_str(
                    f'[Avg.loss: {avg_loss:.6f} loss: {loss.item():.6f} '
                    f'samples: {count_samples}/{amount_samples}] '
                    f'[{self.format_verbose_infos(verbose_infos)}]',
                    refresh=False
                    )
                pbar.update()
                # pbar.refresh()
                if ((ibatch+1) % visualization_interval == 0):
                    self.log_infos(
                        self.current_epoch,
                        ibatch,
                        loss.item(),
                        avg_loss=avg_loss,
                        **verbose_infos
                        )
                    print()
            pbar.close()

    def train_model(
            self,
            dataloader_train: DataLoader,
            dataloader_test: DataLoader = None,
            batch_accumulation=None,
            schedule_pattern=None,
            visualization_interval=10,
            visualization_amount=None,
            checkpoint_interval: int = None,
            checkpoint_filepath: str = None,
            is_drop_last=False,
            **kwds
        ):
        tqdm._instances.clear()
        batch_accumulation = batch_accumulation or self.batch_accumulation
        schedule_pattern = schedule_pattern or self.schedule_pattern
        schedule = self.generate_train_test_schedule(schedule_pattern)
        self.log = []
        self.work_start_time = datetime.now().strftime("%Y%m%d%H%M%S")
        if None is not checkpoint_filepath:
            log_filepath = Path(
                checkpoint_filepath
                ) / f'{self.model.__class__.__name__}_{self.work_start_time}' / f'{self.model.__class__.__name__}_{self.work_start_time}.log'
            self.log_filepath = log_filepath
            self.write_log()
            checkpoint_filepath = Path(
                checkpoint_filepath
                ) / f'{self.model.__class__.__name__}_{self.work_start_time}'

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            print(
                f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] '
                f'Epoch {self.current_epoch+1:02}'
                f' / {self.num_epochs}'
                '\n'
                '------------------------------------------------------------'
                )

            for mode in schedule[epoch]:
                if mode == 'train':
                    self.train_epoch(
                        dataloader_train=dataloader_train,
                        batch_accumulation=batch_accumulation,
                        visualization_interval=visualization_interval,
                        visualization_amount=visualization_amount,
                        is_drop_last=is_drop_last,
                        )
                elif mode == 'test':
                    if dataloader_test is not None:
                        self.test_epoch(
                            dataloader_test=dataloader_test,
                            visualization_interval=visualization_interval,
                            visualization_amount=visualization_amount,
                            is_drop_last=is_drop_last,
                            )
            if None is not checkpoint_filepath:
                Path(str(checkpoint_filepath)).mkdir(
                    parents=True, exist_ok=True
                    )
                if (None is not checkpoint_interval
                   ) and ((epoch+1) % checkpoint_interval == 1):
                    Networker.save_model(self, checkpoint_filepath)
                    print('[Info] Checkpoint saved')
                elif epoch == self.num_epochs - 1:
                    Networker.save_model(self, checkpoint_filepath)
                    print('[Info] Checkpoint saved')

            print(
                '------------------------------------------------------------'
                '\n'
                )

    def pred(self, n_Xs) -> torch.Tensor:
        self.mode = 'test'
        with torch.no_grad():
            n_Xs, _ = self.runtime_fn(n_Xs, None)
            n_Xs = n_Xs.to(self.device)
            preds = self.model(n_Xs)
            if isinstance(preds, tuple):
                preds = [pred.detach().cpu() for pred in preds]
            elif isinstance(preds, torch.Tensor):
                preds = preds.detach().cpu()
        return preds

    def log_infos(self, epochs, ibatch, loss, is_write_log=True, **kwds):
        '''
        记录日志信息，包括当前轮数、当前批次、当前损失等
        '''
        log = {
            'mode': self.mode,
            'epoch': epochs,
            'ibatch': ibatch,
            'loss': loss,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            **kwds,
            }
        self.log.append(log)
        if is_write_log and self.log_filepath:
            self.write_log()

    def write_log(self, filepath: Union[str, Path] = None):
        '''
        将日志信息写入文件
        '''
        filepath = filepath or self.log_filepath
        if isinstance(filepath, str):
            filepath = Path(filepath)
        if len(self.log) == 0:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text('', encoding='utf-8')
            return
        else:
            with filepath.open('a', encoding='utf-8') as f:
                f.write(
                    json.dumps(self.log[-1], ensure_ascii=False) + '\n'
                    )

    def get_lr_infos(self):
        return {
            f'{str(optim_reference["optimizer"].__module__).split(".")[-1]}[{_i}.{_j}]':
                f'{param_group["lr"]:.6f}'
            for _i, optim_reference in enumerate(self.optim_references)
            for _j, param_group in
            enumerate(optim_reference['optimizer'].param_groups)
            }

    def format_verbose_infos(self, verbose_infos: dict[str]):
        if None is verbose_infos:
            return ''
        else:
            _str = ' | '.join([
                f'{k}: {v}' for k, v in verbose_infos.items()
                ]).replace('\n', ' ')
            return _str

    def generate_train_test_schedule(
            self, pattern, num_epochs: int = None
        ):
        """
        生成与实际训练轮次匹配的训练测试计划
        
        Parameters
        ---
        pattern : 
            训练测试模式，例如 [('train', 3), ('test', 1)]
        num_epochs :
            总训练轮次数
        
        Returns
        ---
        列表，每个元素是一个列表，表示该轮次要执行的操作，例如 [['train'], ['train'], ['train', 'test']]
        """
        schedule = []
        current_epoch = 0
        num_epochs = num_epochs or self.num_epochs
        schedule_expand = []
        len_schedule_expand = 0
        for mode, epochs in pattern:
            if mode == 'train':
                len_schedule_expand += epochs
            schedule_expand.extend([mode] * epochs)
        cur_schedule = [schedule_expand[0]]
        for mode in (schedule_expand *
                        (num_epochs//len_schedule_expand + 1))[1:]:
            if current_epoch >= num_epochs:
                schedule = schedule[:current_epoch]
                break
            if mode == 'train':
                current_epoch += 1
                schedule.append(cur_schedule)
                cur_schedule = ['train']
            elif mode == 'test':
                cur_schedule.append('test')
        if 'test' not in schedule[-1]:
            schedule[-1].append('test')
        return schedule

    @staticmethod
    def setup_seed(seed):
        import os
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现

    @staticmethod
    def save_model(
            networker: "Networker",
            filepath: Union[str, Path],
            addition=''
        ):
        """
        Notes
        ---
        保存模型的参数，不保存模型的结构。
        建议只输入工作文件夹而不指定文件名
        """
        model_name = networker.model.__class__.__name__
        cur_epoch = networker.current_epoch
        work_start_time = networker.work_start_time
        cur_time = datetime.now().strftime('%Y%m%d%H%M%S')
        num_epochs = networker.num_epochs
        if 'avg_loss' in networker.log[-1]:
            addition += f'[avg_loss={networker.log[-1]["avg_loss"]:<.3f}]'
        latest_loss = networker.log[-1]['loss']
        filename = f'{model_name}_{work_start_time}_[eph={cur_epoch+1}of{num_epochs}]_[loss={latest_loss:<.3f}]_{addition}.pth'
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
        torch.save(networker.model.state_dict(), str(filepath))

    @staticmethod
    def load_model(networker: "Networker", filepath: Union[str, Path]):
        networker.model.load_state_dict(
            torch.load(filepath, map_location=networker.device)
            )
