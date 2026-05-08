import numpy as np
import cv2

from pathlib import Path
from typing import Literal, Union

from scipykit.utils import *


class Frustum():
    """
    一个运行时用于便捷提供各类处理的数据类。
    
    以帧名为基础，可以快速获取指定相机的图像、相机工具、点云等数据。

    并且尽量使用原生类型
    """

    def __init__(
            self,
            frustum_depths=np.linspace(0, 80, 50),
            **kwds
        ):
        """
        Parameters
        ----------
        frustum_camera_names :
            'all'
        frustum_density : 
            20
        """
        frustum_params = {
            k[8:]: v
            for k, v in kwds.items() if k.startswith('frustum_')
            }
        self.frustums = self._make_frustum_static(
            depths=frustum_depths, **frustum_params
            )

    def _make_frustum_static(
            self,
            depths: np.ndarray,
            camera_names: Union[Literal['all'], list[str]] = 'all',
            density: int = 20,
            **kwds
        ) -> np.ndarray:
        """
        创建相机视锥体

        Parameters
        ----------
        depths : np.ndarray
            深度图
        camera_names : list[str]
            相机名，默认为'all'，表示所有相机
        density : int, optional
            稠密程度，默认为20

        Returns
        -------
        np.ndarray
            视锥体图像
        """
        self.frustums: dict[str, np.ndarray] = {}
        if 'all' == camera_names:
            camera_names = list(self.camera_tools.keys())
        tmp_shape = {self.imgs[k].shape: [] for k in camera_names}
        for camera_name in camera_names:
            tmp_shape[self.imgs[camera_name].shape].append(camera_name)
        for shape, cams in tmp_shape.items():
            coors = CameraTools3D.generate_frustum(
                np.linspace(0, shape[1] - 1, density),  # w
                np.linspace(0, shape[0] - 1, density),  # h
                depths=depths  # d
                )
            for cam in cams:
                self.frustums[cam] = coors
                self.cameras[cam]['frustum'] = coors
        return self.frustums

    def __repr__(self):
        _str = f'DataSample(frame_id={self.frame_id})'
        _str += f'\n\timgs: { {k:v.shape for k,v in self.imgs.items()} }'
        _str += f'\n\tcamera_tools: {list(self.camera_tools.keys())}'
        return _str
