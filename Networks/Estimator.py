from scipykit.utils import *
from pathlib import Path
import torch
from torch.nn import functional as F
from torch import nn
from torch.utils.data import dataset, DataLoader, Subset
import random


# region EM
class EM():

    DEVICE = torch.device("cuda:0")

    def __init__(self, model_cls, **kwds) -> None:
        '''
        Parameters
        ---
        model:
            - 网络
        batch_size:
            - 批大小
        dataloader_train:
            - 训练集
        dataloader_test:
            - 测试集
        loss_fn:
            - 损失函数
        optimizer_cls:
            - 优化器类
        num_epochs:
            - 训练轮数
        '''
        self.model_cls = model_cls
        self.model: nn.Module = self.model_cls()
        self.batch_size = 512
        self.dataloader_train = None
        self.dataloader_test = None
        self.loss_fn = nn.MSELoss()
        self.optimizer_cls = torch.optim.AdamW
        self.log = {}
        self.num_epochs = None
        for k, v in kwds.items():
            if k in self.__dict__:
                self.__dict__[k] = v

        self.model.to(EM.DEVICE)
        self.optimizer = self.optimizer_cls(self.model.parameters())

    def create_dataloader(self, dataset_train, dataset_test):
        self.dataloader_train = DataLoader(
            dataset_train, batch_size=self.batch_size
            )
        self.dataloader_test = DataLoader(
            dataset_test, batch_size=self.batch_size
            )

    def reset_net(self):
        '''
        通过重新生成一个Net来重置网络
        '''
        self.model = self.model_cls()
        self.model.to(EM.DEVICE)
        self.optimizer = self.optimizer_cls(self.model.parameters())

    # region EM.Train
    def train(self):
        print('train', '=' * 18)
        arr_loss = []
        amount_batches = len(self.dataloader_train)
        size_data = len(self.dataloader_train.dataset)
        self.model.train()
        # self.net.eval()
        for ibatch, (n_imgs, n_feats) in enumerate(self.dataloader_train):
            n_imgs: torch.Tensor = n_imgs.to(EM.DEVICE)
            n_feats: torch.Tensor = n_feats.to(EM.DEVICE)
            pred_nfeats = self.model(n_imgs, n_feats)
            # pred_nfeats = self.model(n_imgs)
            train_loss: torch.Tensor = self.loss_fn(
                pred_nfeats, n_feats.to(torch.float32)
                )
            num_train_loss = train_loss.item()
            arr_loss.append(num_train_loss)

            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()

            if ibatch % (amount_batches//4) == 0:
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
            for ibatch, (n_imgs, n_feats) in enumerate(self.dataloader_test
                                                       ):
                n_imgs: torch.Tensor = n_imgs.to(EM.DEVICE)
                n_feats: torch.Tensor = n_feats.to(EM.DEVICE)
                pred_nfeats = self.model(n_imgs, n_feats)
                # pred_nfeats = self.model(n_imgs)
                test_loss = self.loss_fn(
                    pred_nfeats, n_feats.to(torch.float32)
                    )
                num_test_loss = test_loss.item()
                arr_loss.append(num_test_loss)
                if ibatch % (amount_batches//2) == 0:
                    # if ibatch % (2) == 0:
                    print(
                        f'{ibatch*self.batch_size:<5}/{size_data:>5} loss: {num_test_loss}'
                        )
        self.log['test_loss'].append(np.mean(arr_loss))
        print(f'Avg. loss: {np.mean(arr_loss)}')

    def train_model(self, num_epochs: int, lr_fn=lambda _: 0.001):
        EM.setup_seed(0)
        self.num_epochs = num_epochs
        self.log['train_loss'] = []
        self.log['test_loss'] = []
        for t in range(num_epochs):
            print(f"\nEpoch {t+1}\n-------------------------------")
            lr = lr_fn(t / num_epochs)
            self.optimizer.param_groups[0]['lr'] = lr
            print(f'lr: {lr:.6f}')
            self.train()
            self.test()
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

    def pred(self, input: np.ndarray) -> np.ndarray:
        input = torch.tensor(input, dtype=torch.float32).to(EM.DEVICE)
        self.model.eval()
        with torch.no_grad():
            pred: torch.Tensor = self.model(input)
        return pred.detach().cpu().numpy()

    @staticmethod
    def save_model(em: "EM", filepath: str):
        if not str(filepath).endswith('.pkl'):
            filepath = Path(
                filepath
                ) / f'ckpt[num_eopchs={em.num_epochs},dim_feat={em.model.dim_feat};dim_word={em.model.dim_word},len_src={em.model.len_src},len_tgt={em.model.len_tgt},dim_ff={em.model.dim_ff},shift_right={em.model.shift_right},].pkl'
        pkl_save(em.model, filepath)

    @staticmethod
    def load_model(em: "EM", filepath: str):
        model = pkl_load(filepath)
        em.model = model

    # endregion


# region EM
class EM2():

    DEVICE = torch.device("cuda:0")

    def __init__(self, model_cls, **kwds) -> None:
        '''
        Parameters
        ---
        model:
            - 网络
        batch_size:
            - 批大小
        dataloader_train:
            - 训练集
        dataloader_test:
            - 测试集
        loss_fn:
            - 损失函数
        optimizer_cls:
            - 优化器类
        num_epochs:
            - 训练轮数
        '''
        self.model_cls = model_cls
        self.model: nn.Module = self.model_cls()
        self.batch_size = 512
        self.dataloader_train = None
        self.dataloader_test = None
        self.loss_fn = nn.MSELoss()
        self.optimizer_cls = torch.optim.AdamW
        self.log = {'train_loss': [], 'test_loss': []}
        self.num_epochs = None
        for k, v in kwds.items():
            if k in self.__dict__:
                self.__dict__[k] = v

        self.model.to(EM.DEVICE)
        self.optimizer = self.optimizer_cls(self.model.parameters())

    def create_dataloader(self, dataset_train, dataset_test):
        self.dataloader_train = DataLoader(
            dataset_train, batch_size=self.batch_size
            )
        self.dataloader_test = DataLoader(
            dataset_test, batch_size=self.batch_size
            )

    def reset_net(self):
        '''
        通过重新生成一个Net来重置网络
        '''
        self.model = self.model_cls()
        self.model.to(EM.DEVICE)
        self.optimizer = self.optimizer_cls(self.model.parameters())

    # region EM.Train
    def train(self):
        print('train', '=' * 18)
        arr_loss = []
        amount_batches = len(self.dataloader_train)
        size_data = len(self.dataloader_train.dataset)
        self.model.train()
        # self.net.eval()
        for ibatch, (n_imgs, n_feats) in enumerate(self.dataloader_train):
            n_imgs: torch.Tensor = n_imgs.to(EM.DEVICE)
            n_feats: torch.Tensor = n_feats.to(EM.DEVICE)
            pred_nfeats = self.model(n_imgs)
            # pred_nfeats = self.model(n_imgs)
            train_loss: torch.Tensor = self.loss_fn(pred_nfeats, n_feats)
            num_train_loss = train_loss.item()
            arr_loss.append(num_train_loss)

            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()

            if ibatch % (amount_batches//4) == 0:
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
            for ibatch, (n_imgs, n_feats) in enumerate(self.dataloader_test
                                                       ):
                n_imgs: torch.Tensor = n_imgs.to(EM.DEVICE)
                n_feats: torch.Tensor = n_feats.to(EM.DEVICE)
                pred_nfeats = self.model(n_imgs)
                # pred_nfeats = self.model(n_imgs)
                test_loss = self.loss_fn(pred_nfeats, n_feats)
                num_test_loss = test_loss.item()
                arr_loss.append(num_test_loss)
                if ibatch % (amount_batches//2) == 0:
                    # if ibatch % (2) == 0:
                    print(
                        f'{ibatch*self.batch_size:<5}/{size_data:>5} loss: {num_test_loss}'
                        )
        self.log['test_loss'].append(np.mean(arr_loss))
        print(f'Avg. loss: {np.mean(arr_loss)}')

    def train_model(self, num_epochs: int, lr_fn=lambda _: 0.001):
        EM.setup_seed(0)
        self.num_epochs = num_epochs
        self.log['train_loss'] = []
        self.log['test_loss'] = []
        for t in range(num_epochs):
            print(f"\nEpoch {t+1}\n-------------------------------")
            lr = lr_fn(t / num_epochs)
            self.optimizer.param_groups[0]['lr'] = lr
            print(f'lr: {lr:.6f}')
            self.train()
            self.test()
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

    def pred(self, input: np.ndarray, dtype=torch.float32) -> np.ndarray:
        input = torch.tensor(input, dtype=dtype).to(EM.DEVICE)
        self.model.eval()
        with torch.no_grad():
            pred: torch.Tensor = self.model(input)
        return pred.detach().cpu().numpy()

    @staticmethod
    def save_model(em: "EM", filepath: str):
        if not str(filepath).endswith('.pkl'):
            # filepath = Path(
            #     filepath
            #     ) / f'ckpt[num_eopchs={em.num_epochs},dim_feat={em.model.dim_feat};dim_word={em.model.dim_word},len_src={em.model.len_src},len_tgt={em.model.len_tgt},dim_ff={em.model.dim_ff},shift_right={em.model.shift_right},].pkl'
            filepath = Path(
                filepath
                ) / f'ckpt[num_eopchs={em.num_epochs}].pkl'
        pkl_save(em.model, filepath)

    @staticmethod
    def load_model(em: "EM", filepath: str):
        model = pkl_load(filepath)
        em.model = model

    # endregion
