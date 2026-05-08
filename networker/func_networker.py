from torch.optim.adamw import AdamW
from scipykit.utils import *
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import dataset, DataLoader, Subset
import random

from typing import Literal, Union

from .base_networker import BaseNetworker


class FuncNetworker(BaseNetworker):

    def __init__(
            self,
            model_cls,
            batch_size=512,
            num_epochs=100,
            dataloader_train=None,
            dataloader_test=None,
            runtime_fn=lambda s, *x: x,
            model_dtype=torch.float32,
            loss_fn=None,
            optimizer_cls=torch.optim.AdamW,
            optimizer_cfg: dict = None,
            verbose_interval_train=4,
            verbose_interval_test=2,
            checkpoint_interval=None,
            device=None,
            **kwds
        ):
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
        runtime_fn:
            - 执行训练和测试时，需要实时对数据进行的操作
            - 例如，当内存压力过大时，可以本地存储更小体积的类型，在运行时转换为精度更高的类型参与运算
            - 形式为 `fun(self, n_Xs, n_Ys) -> n_Xs, n_Ys`
        '''
        super().__init__(
            model_cls=model_cls,
            batch_size=batch_size,
            num_epochs=num_epochs,
            dataloader_train=dataloader_train,
            dataloader_test=dataloader_test,
            model_dtype=model_dtype,
            loss_fn=loss_fn,
            optimizer_cls=optimizer_cls,
            optimizer_cfg=optimizer_cfg,
            verbose_interval_train=verbose_interval_train,
            verbose_interval_test=verbose_interval_test,
            checkpoint_interval=checkpoint_interval,
            device=device,
            **kwds
            )
        self.runtime_fn = runtime_fn

    def train(self):
        print('train', '=' * 18)
        arr_loss = []
        amount_batches = len(self.dataloader_train)
        size_data = len(self.dataloader_train.dataset)
        self.model.train()
        for ibatch, (n_Xs, n_Ys) in enumerate(self.dataloader_train):
            # print(ibatch)
            n_Xs, n_Ys, = self.runtime_fn(self, n_Xs, n_Ys)
            n_Xs: torch.Tensor = n_Xs.to(self.DEVICE)
            n_Ys: torch.Tensor = n_Ys.to(self.DEVICE)
            n_Ys_pred = self.model(n_Xs)
            train_loss: torch.Tensor = self.loss_fn(n_Ys_pred, n_Ys)
            # print(train_loss)
            # print(self.optimizer.state_dict())
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
                n_Xs, n_Ys, = self.runtime_fn(self, n_Xs, n_Ys)
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

    def pred(
            self,
            input: torch.Tensor,
            dtype=torch.float32
        ) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            input, _ = self.runtime_fn(self, input, None)
            input = input.to(self.DEVICE)
            preds = self.model(input)
            if isinstance(preds, tuple):
                preds = [pred.detach().cpu() for pred in preds]
            elif isinstance(preds, torch.Tensor):
                preds = preds.detach().cpu()
        return preds
