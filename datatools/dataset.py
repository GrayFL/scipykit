from scipykit.utils import *
from pathlib import Path
import pandas as pd
import torch
from torch import nn
from torch.utils.data import dataset, DataLoader, Subset
import random
from copy import deepcopy, copy

from .pose import Skeleton


# region Dataset
class Dataset(dataset.Dataset):
    '''
    self.mode:
    - `flat`: 所有元素展平成列向量的形式 (-1, 1)
    - `vector`: 向量形式 (-1, 2)
    - `pattern`: 向量对，一左一右 (-1, 4)
    '''

    def __init__(self, rootpath: str = None, **kwds) -> None:
        '''
        Parameters
        ---
        datatype:
            - `football`
            - `coco-wholebody`
        is_need_load_image:
            - 是否需要在创建数据集时导入图像
        which:
            - `val` (Default) 导入验证集
            - `train` 导入训练集
        '''
        super().__init__()
        self.PATH_ROOT: Path = None
        self.PATH_IMG: Path = None
        self.PATH_ANN: Path = None
        self.datatype: str = None
        self.target_shape = (640, 640)
        self.mode = 'vector'
        self.annotation: pd.DataFrame = None
        self.feature = None
        self.images = None
        self.train_index = None
        self.test_index = None
        self.is_unified = {'images': False, 'annotation': False}
        self.get_image = lambda: None
        if None is not rootpath:
            self.load(rootpath, **kwds)

    def load(
            self,
            rootpath: str,
            datatype: str = None,
            is_need_load_image=False,
            which='val'
        ):
        if 'football' == datatype:
            self.datatype = datatype
            self.PATH_ROOT = Path(rootpath)
            self.PATH_IMG = self.PATH_ROOT / 'images'
            self.PATH_ANN = self.PATH_ROOT / 'annotations.json'
            self.datatype = datatype
            raw = json_load(self.PATH_ANN)
            raw['image_shape'] = [(h, w) for (w, h) in raw['image_shape']]
            raw['keypoints'] = [np.array(pts) for pts in raw['keypoints']]
            self.annotation = pd.DataFrame(raw)

            def get_image(image_id: int):
                img_path = self.PATH_IMG / f'im{image_id:>04}.jpg'
                img = cv2_imread(img_path)
                return img

            self.get_image = get_image

            if is_need_load_image:
                self.images = self.load_images()
        if 'coco-wholebody' == datatype:
            self.datatype = datatype
            self.PATH_ROOT = Path(rootpath)
            self.PATH_IMG = self.PATH_ROOT / f'{which}2017'
            self.PATH_ANN = self.PATH_ROOT / f'coco_wholebody_{which}_v1.0.json'
            raw = json_load(self.PATH_ANN)
            dic_img_info = {}
            for img_info in raw['images']:  # 这里记录了每个图片的id和对应尺寸
                dic_img_info[img_info['id']
                             ] = (img_info['height'], img_info['width'])
            df = pd.DataFrame(raw['annotations'])[[
                'id', 'image_id', 'keypoints', 'num_keypoints'
                ]]
            df['keypoints'] = df['keypoints'].apply(
                lambda d: np.array(d, dtype=np.float32).reshape(-1, 3)[:, :
                                                                        2]
                )
            df['image_shape'] = df['image_id'].apply(
                lambda d: dic_img_info[d]
                )
            self.annotation = df

            def get_image(image_id: int):
                img_path = self.PATH_IMG / f'{image_id:>012}.jpg'
                img = cv2_imread(img_path)
                return img

            self.get_image = get_image

            if is_need_load_image:
                self.images = self.load_images()

    def load_images(self):
        images = {}
        for image_id in self.annotation['image_id']:
            images[image_id] = self.get_image(image_id)
        self.images = images
        return images

    @staticmethod
    def unify_image(image: np.ndarray, target_shape: tuple[int, int]):
        # 获取图像的尺寸
        h, w = image.shape[:2]
        target_h, target_w = target_shape
        # 计算缩放比例
        if h > w:
            scale = target_h / h
        else:
            scale = target_w / w
        # 缩放图像
        resized_image = cv2.resize(image, None, fx=scale, fy=scale)
        # 获取新的尺寸
        new_h, new_w = resized_image.shape[:2]
        # 创建目标尺寸的背景
        canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 117
        # 将缩放后的图像放置在中心
        start_h = (target_h - new_h) // 2
        start_w = (target_w - new_w) // 2
        canvas[
            start_h:start_h + new_h,
            start_w:start_w + new_w,
            ] = resized_image
        return canvas

    @staticmethod
    def map_points_to_target(
            points,
            original_shape: tuple[int, int],
            target_shape: tuple[int, int],
            padding_mode='center'
        ):
        '''
        将原始图像上的一系列点映射到目标图像上的对应点，考虑填充过程

        Parameters:
        ---
        points: 原始图像上的点列表，形如 [[x1, y1], [x2, y2], ...]
        original_shape: 原始图像的尺寸，(height, width)
        target_shape: 目标图像的尺寸，(height, width)
        padding_mode: 填充模式，可以是 'center'（默认）或 'edge'

        Return:
        ---
        target_points: 目标图像上对应的点列表，形如 [[tx1, ty1], [tx2, ty2], ...]
        '''
        original_h, original_w = original_shape
        target_h, target_w = target_shape
        # 计算缩放比例
        if original_h > original_w:
            scale = target_h / original_h
        else:
            scale = target_w / original_w
        # 根据填充模式调整目标图像的起始坐标
        if padding_mode == 'center':
            offset_x = (target_w - original_w * scale) / 2
            offset_y = (target_h - original_h * scale) / 2
        elif padding_mode == 'edge':
            offset_x = 0
            offset_y = 0
        else:
            raise ValueError(
                "Unsupported padding mode. Use 'center' or 'edge'."
                )
        # 对每个点进行映射
        target_points = points * scale + np.array([offset_x, offset_y])
        return target_points

    def unify(self):
        '''
        将图像和关键点进行归一化
        '''
        if ((None is not self.images)
            and False == self.is_unified['images']):
            for img_id in self.images.keys():
                self.images[img_id] = Dataset.unify_image(
                    self.images[img_id], self.target_shape
                    )
            self.is_unified['images'] = True
        if False == self.is_unified['annotation']:
            self.annotation['keypoints'] = self.annotation.loc[:, [
                'keypoints', 'image_shape'
                ]].apply(
                    lambda row: Dataset.map_points_to_target(
                        row['keypoints'],
                        row['image_shape'],
                        self.target_shape,
                        ),
                    axis=1
                    )
            self.is_unified['annotation'] = True

    def generate_feature(self, mode: str, is_need_unify=False):
        '''
        !Warning: 请在使用该函数前确保已经进行了Unify操作

        Parameters
        ---
        mode:
            - `pts2flat`: 所有元素展平成列向量的形式 (-1, 1)
            - `pts2vector`: 向量形式 (-1, 2)
            - `pts2pattern`: 向量对，一左一右 (-1, 4)
            - 反向同理，如 `vector2pts`
        '''
        if is_need_unify:
            self.unify()
        self.annotation['feature'] = self.annotation['keypoints'].apply(
            lambda d: Skeleton.
            cvt(d, self.datatype, mode, image_shape=self.target_shape)
            )
        self.feature = self.annotation['feature'].values

    def filter_dataset(
            self, func=lambda: None, inplace=False
        ) -> "Dataset":
        '''
        从原始数据中摘选出一些数据生成总数据集
        !warning: 只过滤annotation，不会改变feature，所以记得generate_feature()

        Parameters
        ---
        func: 判别表达式，入参为dataframe的每一行的值，返回值为T/F `lambda x: x['a'] > 2`
        '''
        sub_ann = self.annotation[self.annotation.apply(func, axis=1)
                                  ].reset_index(drop=True)
        if inplace:
            self.annotation = sub_ann
        else:
            ret = deepcopy(self)
            ret.annotation = sub_ann
            return ret

    def split_dataset(self, train_ratio=0.8, mode='subset'):
        '''
        mode:
            - `subset` 输出为Subset类型
            - `self` 输出为完整的self
        '''
        div_index = int(train_ratio * len(self))
        arr_index = np.random.permutation(len(self))
        self.train_index = arr_index[:div_index]
        self.test_index = arr_index[div_index:]
        if 'subset' == mode:
            subset_train = Subset(self, self.train_index)
            subset_test = Subset(self, self.test_index)
        elif 'self' == mode:
            ...
        return subset_train, subset_test

    def feat2quiver(
        self,
        feat: np.ndarray,
        style='basic',
        ):
        '''
        feat:
            - `self.mode == vector` (-1, 2)
            这个mode是数据集的mode，而非skeleton的
        mode:
            - `basic`: [肩中, 髋中, *骨骼(量化)]
        Return
        ---
        quiver, pts, bones
        '''
        if 'vector' == self.mode and 'basic' == style:
            pts, bones = Skeleton.cvt(
                feat,
                self.datatype,
                'vector2pts',
                image_shape=self.target_shape
                )
            quiver = np.hstack(
                [
                    pts[[2, 1, 1, 1, 3, 4, 5, 6, 2, 2, 9, 10, 11, 12]],
                    bones,
                    ],
                dtype=np.float32,
                )
            return quiver, pts, bones
        if 'vector' == self.mode and 'relative' == style:
            pts, bones = Skeleton.cvt(
                feat, self.datatype, 'vector2pts', style
                )
            quiver = np.hstack(
                [
                    pts[[2, 1, 1, 1, 3, 4, 5, 6, 2, 2, 9, 10, 11, 12]],
                    bones,
                    ],
                dtype=np.float32,
                )
            return quiver, pts, bones

    def __len__(self):
        return self.annotation.__len__()

    def __getitem__(self, index):
        if 'vector' == self.mode:
            feat = self.feature[index]
            return feat[0:10], feat[10:16]

    def restore(self, X: np.ndarray, Y: np.ndarray):
        '''
        从之前分割的量获得feature
        '''
        if 'vector' == self.mode:
            feat = np.concatenate([X, Y], axis=0)
            return feat

    def __repr__(self) -> str:
        return self.annotation.__repr__()

    # endregion
