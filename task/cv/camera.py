import numpy as np
import cv2
import time
# from simple_detector import detect, pose_detect, pose_predictor
# from TrackTools import TrackMaster
from scipykit.utils import *
# from pathlib import Path
from typing import Literal, Union


class CameraTools3D():

    def __init__(
            self,
            intrinsics: np.ndarray = None,
            intrinsics_ext: np.ndarray = None,
            distortion: np.ndarray = None,
            img_shape: Union[list, tuple, np.ndarray] = None,
            alpha: float = 1.0,
            scale: float = 1.0,
            mtx_obj2cam: np.matrix = None,
            mtx_cam2obj: np.matrix = None
        ) -> None:
        """
        Parameters
        ---
        intrinsics : np.ndarray
            相机内参
        intrinsics_ext : np.ndarray
            正畸后的等效相机内参
        distortion : np.ndarray
            相机畸变
        img_shape : Union[list, tuple, np.ndarray]
            原始图像尺寸 (w, h)
        alpha : float
            即正畸后黑边大小系数
        scale : float
            缩放比例，即最终图像大小是原来的多少倍
        mtx_obj2cam : np.matrix
            物理坐标到相机坐标的投影矩阵
        mtx_cam2obj : np.matrix
            相机坐标到物理坐标的投影矩阵
        mapx : np.ndarray
            畸变矫正映射表，src_x = mapx[dst_y, dst_x]
        mapy : np.ndarray
            畸变矫正映射表，src_y = mapy[dst_y, dst_x]
        """
        if ((None is not intrinsics)
                and not isinstance(intrinsics, np.matrix)):
            intrinsics = np.matrix(intrinsics)
        self.intrinsics: np.matrix = intrinsics  # 内参
        if ((None is not intrinsics_ext)
                and not isinstance(intrinsics_ext, np.matrix)):
            intrinsics_ext = np.matrix(intrinsics_ext)
        self.intrinsics_ext: np.matrix = intrinsics_ext  # 内参
        # self.extrinsics:np.ndarray=... # 外参
        if ((None is not distortion)
                and not isinstance(distortion, np.ndarray)):
            distortion = np.array(distortion)
        self.distortion: np.ndarray = distortion  # 畸变
        self.err = ...
        self.mapx: np.ndarray = ...
        self.mapy: np.ndarray = ...
        self.img_shape = img_shape
        self.alpha = alpha
        self.scale = scale
        # self.perspective_trans_mtx:np.ndarray=... # 透视变换矩阵
        # self.perspective_trans_mtx_inv:np.ndarray=... # 透视变换矩阵

        if (None is not mtx_obj2cam) or (None is not mtx_cam2obj):
            if None is not mtx_obj2cam:
                mtx_obj2cam = np.matrix(mtx_obj2cam)
                mtx_cam2obj = mtx_obj2cam.I
            else:
                mtx_cam2obj = np.matrix(mtx_cam2obj)
                mtx_obj2cam = mtx_cam2obj.I
            # 如果提供了投影矩阵，则直接使用
        self.mtx_obj2cam: np.matrix = mtx_obj2cam
        self.mtx_cam2obj: np.matrix = mtx_cam2obj

    def calibrate_camera(
            self,
            arr_img: 'list[np.ndarray]',  # 用于标定的原始图像
            obj_point: np.ndarray = ...,  # 棋盘格的绝对位置
            pattern_size=[7, 7],  # 棋盘格角点尺寸
            block_size: float = 1000 / 8,  # 期盘格的格子大小
            alpha: float = None,
            scale: float = None
        ):
        '''
        注意img是三通道的

        Parameters
        ---
        pattern_size : list[int]
            [h,w]
        '''
        self.alpha = alpha or self.alpha  # 若未指定则使用初始化时的alpha
        self.scale = scale or self.scale  # 若未指定则使用初始化时的scale

        arr_imgp = []
        arr_objp = []
        objp = np.zeros([pattern_size[0] * pattern_size[1], 3],
                        dtype=np.float32)
        l1, r1 = -(pattern_size[0] // 2), (pattern_size[0] + 1) // 2
        # 7 -> [-3,4]
        l2, r2 = -(pattern_size[1] // 2), (pattern_size[1] + 1) // 2
        objp[:, :2] = np.mgrid[l1:r1, l2:r2].transpose(2, 1, 0).reshape([
            -1, 2
            ]) * block_size
        objp[:, 1] = -objp[:, 1]
        shape = np.array(arr_img[0].shape[-2:-4:-1])
        if None is scale:
            dst_shape = None
        else:
            dst_shape = (scale * shape).astype(np.int16)
        for i, img in enumerate(arr_img):
            ret, imgp = cv2.findChessboardCornersSB(img, pattern_size)
            # print(ret,imgp)
            if not ret:
                print(f'{i:>03}  omit...')
                continue
            imgp = imgp.transpose(1, 0, 2)[0]
            arr_imgp.append(imgp)
            arr_objp.append(objp)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            arr_objp, arr_imgp, shape, None, None
            )
        self.intrinsics = mtx
        self.distortion = dist
        self.err = ret

        newMtx, (x, y, w, h) = cv2.getOptimalNewCameraMatrix(
            cameraMatrix=mtx,
            distCoeffs=dist,
            imageSize=shape,
            alpha=alpha,
            newImgSize=dst_shape,
            centerPrincipalPoint=True
            )
        # self.img_shape = dst_shape
        self.intrinsics_ext = np.matrix(newMtx)
        # newMtx[1,1] = newMtx[0,0]
        mapx, mapy = cv2.initUndistortRectifyMap(
            cameraMatrix=mtx,
            distCoeffs=dist,
            R=None,
            newCameraMatrix=newMtx,
            size=[2 * x + w, 2 * y + h],
            m1type=cv2.CV_32FC1
            )
        if None is scale:
            self.scale = self.mapx.shape[0] / shape[0]
        else:
            self.mapx = cv2.resize(mapx, dst_shape)
            self.mapy = cv2.resize(mapy, dst_shape)
        return ret

    def update(self, **kwd):
        """
        Parameters
        ---
        intrinsics : np.ndarray
            相机内参
        intrinsics_ext : np.ndarray
            正畸后的等效相机内参
        distortion : np.ndarray
            相机畸变
        img_shape : Union[list, tuple, np.ndarray]
            原始图像尺寸
        alpha : float
            即正畸后黑边大小系数
        scale : float
            缩放比例，即最终图像大小是原来的多少倍
        mtx_obj2cam : np.matrix
            物理坐标到相机坐标的投影矩阵
        mtx_cam2obj : np.matrix
            相机坐标到物理坐标的投影矩阵
        """
        for k, v in kwd.items():
            if (None is not v) and (k in self.__dict__.keys()):
                self.__dict__[k] = v
                # print(f'update {k} with {v}, now {self.__dict__[k]}')
        img_shape = np.array(self.img_shape, dtype=np.int16)
        alpha = self.alpha
        scale = self.scale
        if (scale > 0):
            dst_shape = (scale * img_shape).astype(np.int16)
        else:
            dst_shape = None

        newMtx, (x, y, w, h) = cv2.getOptimalNewCameraMatrix(
            cameraMatrix=self.intrinsics,
            distCoeffs=self.distortion,
            imageSize=img_shape,
            alpha=alpha,
            newImgSize=dst_shape,
            centerPrincipalPoint=True
            )
        self.intrinsics_ext = np.matrix(newMtx)
        # newMtx 是正畸后的最佳相机内参

        mapx, mapy = cv2.initUndistortRectifyMap(
            cameraMatrix=self.intrinsics,
            distCoeffs=self.distortion,
            R=None,
            newCameraMatrix=newMtx,
            # size=[2 * x + w, 2 * y + h],
            # size=[w,h],
            size=dst_shape,
            m1type=cv2.CV_32FC1
            )
        if (scale > 0):
            tune_scale = CameraTools3D.tune_scale(
                scale, img_shape, mapx.shape[::-1]
                )
            # mapx = cv2.resize(mapx, None, fx=tune_scale, fy=tune_scale)
            # mapy = cv2.resize(mapy, None, fx=tune_scale, fy=tune_scale)
        self.mapx = mapx
        self.mapy = mapy
        return self

    def get_undistort_img(self, img):
        """
        Parameters
        ----------
        img : np.ndarray
            原始图像
        """
        return cv2.remap(
            img,
            self.mapx,
            self.mapy,
            cv2.INTER_LINEAR,
            borderValue=[128, 128, 128]
            # borderValue=[0, 0, 0]
            )

    def cvt_coors(
            self,
            coors: Union[list[np.ndarray], np.ndarray],
            # depths: Union[list[np.number], np.number, np.ndarray] = None,
            mode: Literal[
                'img2obj',
                'obj2img',
                'obj2cam',
                'cam2img',
                'img2cam',
                'cam2img',
                ] = None,
            **kwds
        ):
        """
        转换坐标

        Parameters
        ---
        coors : np.ndarray
            图像坐标或物理坐标  
            其中，图像坐标是 [[w, h]]^T ，物理坐标是 [[x, y, z]]^T
            注意！默认为列向量  
            当批量转换时，可以将坐标组织为列表或三维数组  
            shape [n, 2, 1] or [n, 3, 1]
        mode : Literal['img2obj', 'obj2img']
            转换模式
        """
        if isinstance(coors, list):
            coors = np.array(coors)
        if coors.ndim <= 2:
            coors = coors[None, ...]
        assert coors.ndim == 3, 'coors 输入维度错误'
        assert coors[0].shape[-1] == 1, '输入不是列向量 coors must be column vector'
        coors = coors.astype(np.float32)

        # if None is not depths:
        #     depths = np.array(depths)
        #     if depths.ndim == 0:
        #         depths = depths[None, ...]
        #     depths = depths[..., None, None]
        #     assert depths.ndim == 3, 'depths 输入维度错误'

        intrinsics_ext = np.matrix(
            np.concatenate(
                [self.intrinsics_ext, np.zeros([3, 1])],
                axis=1,
                )
            )
        if 'obj2cam' == mode:
            # 齐次化
            coors = np.concatenate(
                [coors, np.ones([coors.shape[0], 1, 1])],
                axis=1,
                )
            coors_cam = np.asarray(
                (self.mtx_obj2cam[None, ...] @ coors)[:, :3, None]
                )
            return coors_cam

        elif 'cam2img' == mode:
            tmp = np.asarray((self.intrinsics_ext @ coors)[..., None])
            coors_img = tmp[:, :2, :] / tmp[:, [2], :]
            return coors_img

        elif 'obj2img' == mode:
            coors_cam = self.cvt_coors(coors, 'obj2cam')
            coors_img = self.cvt_coors(coors_cam, 'cam2img')
            return coors_img

        elif 'img2cam' == mode:
            # assert coors.shape[-2] == 2, '输入不是图像坐标'
            # assert None is not depths, '未指定深度'
            # coors = depths * coors
            if None is not self.distortion:
                undistorted_coors = cv2.undistortPoints(
                    coors[:, :2, ...].transpose(0, 2,
                                                1).astype(np.float64),
                    self.intrinsics,
                    self.distortion,  # R=np.eye(3),
                    P=self.intrinsics_ext
                    ).transpose(0, 2, 1)
                coors[:, :2, ...] = undistorted_coors
            if coors.shape[-2] == 3:
                # 混合深度信息
                depths = coors[:, [2], ...]
                tmp = np.concatenate(
                    [
                        coors[:, :2, :] * depths,
                        depths,
                        ],
                    axis=1,
                    )
            elif coors.shape[-2] == 2:
                depths: np.ndarray = kwds.get('depths')
                if depths.ndim == 1:
                    depths = depths[..., None, None]
                assert depths.ndim == 3, 'depths 输入维度错误'
                tmp = np.concatenate(
                    [
                        coors * depths,
                        depths,
                        ],
                    axis=1,
                    )
            coors_cam = np.asarray(
                (self.intrinsics_ext.I @ tmp)[:, :3, None]
                )
            return coors_cam

        elif 'cam2img' == mode:
            # # coors,_ = cv2.projectPoints(
            # #     coors,
            # #     rvec=np.eye(3),
            # #     tvec=np.zeros((3, 1)),
            # #     cameraMatrix=self.intrinsics,
            # #     distCoeffs=self.distortion,
            # #     )
            # # coors = coors.transpose(0, 2, 1)
            # w = coors[:, 0, 0]
            # h = coors[:, 1, 0]
            # coors = np.concatenate([self.mapx[w, h], self.mapy[w, h]],
            #                         axis=0)
            # return coors
            ...

        elif 'img2obj' == mode:
            if coors.shape[-2] == 3:
                ...
                # 混合深度信息
            # 计算去畸变后的归一化坐标
            undistorted_coors = cv2.undistortPoints(
                coors[:, :2, ...].transpose(0, 2, 1),
                self.intrinsics,
                self.distortion,
                P=self.intrinsics_ext
                ).transpose(0, 2, 1)
            coors[:, 0, ...] = undistorted_coors[:, 0, ...]
            coors[:, 1, ...] = undistorted_coors[:, 1, ...]
            coors[:, :2, ...] = coors[:, :2, ...] * coors[:, [2], ...]
            # 齐次坐标
            tmp = np.concatenate(coors, axis=1)
            tmp = intrinsics_ext.I @ tmp
            # return np.split(tmp[:3, :], coors.shape[0], axis=1)
            # print(tmp, tmp.shape)
            tmp[3, :] = 1
            # print(tmp)
            # 凑成二维矩阵做运算
            tmp = self.mtx_cam2obj @ tmp
            # 计算物理坐标
            # print(tmp)
            # print(tmp.shape)
            coors_obj = np.array(tmp[:3, :])[..., None].transpose(1, 0, 2)
            # coors_obj = np.split(tmp[:3, :], coors.shape[0], axis=1)
            return coors_obj

    @staticmethod
    def tune_scale(scale, img_shape, map_shape):
        """
        采用这个scale将保证输出的尺寸一定不会比原始图片大

        Parameters
        ---
        scale : float
            缩放比例
        img_shape : tuple
            原始图像尺寸，[w, h]
        map_shape : tuple
            映射后的图像尺寸，[w, h]
        """
        dst_shape = scale * np.array(img_shape)
        map_shape = np.array(map_shape)
        scale = np.min(dst_shape / map_shape)
        return scale

    @staticmethod
    def generate_frustum(*xi, depths: np.ndarray = np.array([1])):
        """
        生成三维长方体内部的散点坐标。

        Parameters
        ---
        xi : tuple
            轴的取点。

        Returns
        ---
        np.ndarray :
            形状为 () 的数组，包含所有散点的坐标。
        """

        # 创建三维网格
        arr_X = np.meshgrid(*xi, depths, indexing='ij')

        # 将网格点转换为一维数组
        points = np.vstack([X.flatten() for X in arr_X]).T
        # points = points[:, :2] * points[:, [2]]
        points = points[..., None]
        return points


#     def make_texture_dis(
#             self,
#             tex: np.ndarray,
#             arr_color: np.ndarray,
#             mat_src: np.ndarray,
#             mat_dst: np.ndarray
#         ):
#         '''
#         测距图是1280x1280的 其假定为20m x 20m的真实大小。
#         第一个参数描述的是某矩形在测距图平面上的角点
#         第二个参数表述的是它在实际照片中的角点
#         通过计算得到的投影矩阵可以将测距图投射到现实世界
#         '''
#         # tex_dis = cv2_imread('相机校正/距离贴图_003.png')[:,:,0]
#         # mtx_trans = cv2.getPerspectiveTransform(
#         #     np.array([[-1,-4],[1,-4],
#         #             [-1,-3],[1,-3],],dtype=np.float32)*(1280/20)+640,
#         #     np.array([[745,752],[1171,752],
#         #                 [676,822],[1244,825],],dtype=np.float32),
#         # )
#         self.mtx_tex2img, _ = cv2.findHomography(mat_src, mat_dst)
#         self.texture_dis = cv2.warpPerspective(
#             tex, self.mtx_tex2img, dsize=self.img_shape
#             )
#         self.mtx_img2tex = np.linalg.inv(self.mtx_tex2img)
#         cdict = {}

#         # arr_color = np.array([[0.0, 0, 0], [0.20, 1, 0], [0.70, 1, 0], [1.0, 1, 1]])
#         # for i,marker in enumerate(arr_color[1:]):
#         # y = 1/(0.7-0.2) *(x-0.2) +0
#         # x = (y-0)*(0.7-0.2)/1 +0.2
#         def inv_fun(x1, x2, y1, y2):
#             return lambda v: (v/256 - y1) * (x2-x1) / (y2-y1) + x1

#         for i in range(1, len(arr_color)):
#             print(arr_color[i - 1])
#             # fun = lambda v: (v/256-arr_color[i-1,2])*(arr_color[i,0]-arr_color[i-1,0])/(arr_color[i,1]-arr_color[i-1,2])+arr_color[i-1,0]
#             cdict[i - 1] = {
#                 'marker':
#                     arr_color[i, 0],
#                 'func':
#                     inv_fun(
#                         arr_color[i - 1, 0],
#                         arr_color[i, 0],
#                         arr_color[i - 1, 2],
#                         arr_color[i, 1]
#                         )
#                 }
#             fun = cdict[i - 1]['func']
#             print(fun(0), fun(128), fun(256))
#         self.color_dict = cdict

#     def make_perspective_trans_mtx(
#             self, mat_src: np.ndarray, mat_dst: np.ndarray, option: str
#         ):
#         '''
#         测距图是1280x1280的 其假定为20m x 20m的真实大小。
#         第一个参数描述的是某矩形在测距图平面上的角点
#         第二个参数表述的是它在实际照片中的角点
#         通过计算得到的投影矩阵可以将测距图投射到现实世界

#         Parameters
#         ---
#         option:
#             'tex2img': 材质贴图和图像坐标的转换
#             'obj2img': 物理坐标和图像坐标的转换
#         '''
#         # tex_dis = cv2_imread('相机校正/距离贴图_003.png')[:,:,0]
#         # mtx_trans = cv2.getPerspectiveTransform(
#         #     np.array([[-1,-4],[1,-4],
#         #             [-1,-3],[1,-3],],dtype=np.float32)*(1280/20)+640,
#         #     np.array([[745,752],[1171,752],
#         #                 [676,822],[1244,825],],dtype=np.float32),
#         # )
#         if option == 'tex2img':
#             self.mtx_tex2img, _ = cv2.findHomography(mat_src, mat_dst)
#             self.mtx_img2tex = np.linalg.inv(self.mtx_tex2img)
#         elif option == 'obj2img':
#             self.mtx_obj2img, _ = cv2.findHomography(mat_src, mat_dst)
#             self.mtx_img2obj = np.linalg.inv(self.mtx_obj2img)

#     def set_regression_param(
#             self,
#             coef: np.ndarray = None,
#             intercept: np.ndarray = None,
#             tbl_depth: np.ndarray = None,
#             tbl_pred_x: np.ndarray = None,
#             tbl_pred_y: np.ndarray = None
#         ):
#         '''
#         在外部手动计算深度图的回归系数

#         其中自变量为depth 去预测x和y 一般来说y更准确

#         LinearRegression().fit(X=df[['v_depth']],y=df[['v_xs','v_ys']])

#         ---

#         coef: 项相关系数
#         intercept: 截距
#         '''
#         self.regression_param = {
#             'coef': coef,
#             'intercept': intercept,
#             'tbl_depth': tbl_depth,
#             'tbl_pred_x': tbl_pred_x,
#             'tbl_pred_y': tbl_pred_y,
#             }

#     def get_objpoint_from_depth(self, depth):
#         '''
#         通过深度和之前计算的回归系数求物理坐标

#         ---

#         depth: 深度 [d,...]

#         ---

#         return: [[[x,y]],...]    shape[-1,1,2]
#         '''
#         if not isinstance(depth, np.ndarray):
#             depth = np.array([depth]).reshape([-1, 1])
#         elif len(depth.shape) > 1:
#             depth = depth.reshape([-1, 1])
#         res1 = (
#             depth @ self.regression_param['coef'].T
#             + self.regression_param['intercept']
#             ).reshape(-1, 1, 2)
#         res2 = np.vstack([
#             np.interp(
#                 depth,
#                 self.regression_param['tbl_depth'],
#                 self.regression_param['tbl_pred_x']
#                 ),
#             np.interp(
#                 depth,
#                 self.regression_param['tbl_depth'],
#                 self.regression_param['tbl_pred_y']
#                 )
#             ]).T.reshape(-1, 1, 2)
#         return res2

#     def get_depth_bbox(self, dpt: np.ndarray, xyxys: np.ndarray):
#         '''
#         给出深度图和检测框坐标，直接求得深度

#         采样点是检测框的中心

#         ---

#         xyxys: [[x1,y1,x2,y2],...] or [[w1,h1,w2,h2],...]

#         ---

#         return: [d,...]    shape[-1,1,2]
#         '''
#         if len(xyxys.shape) == 1:
#             xyxys = xyxys.reshape(-1, xyxys.shape[0])

#         y = int(np.mean(xyxys[:, [3, 1]], axis=1))
#         x = int(np.mean(xyxys[:, [0, 2]], axis=1))
#         depths = dpt[y, x]
#         return depths

#     def get_objpoint(self, coors: np.ndarray):
#         '''
#         通过反投影，给出图像坐标计算物理坐标

#         ---

#         coors: [[[x,y]],...] or [[[w,h]],...]

#         ---

#         return: [[[x,y]],...]   shape[-1,1,2]
#         '''
#         if len(coors.shape) <= 2:
#             coors = coors.reshape(-1, 1, coors.shape[-1])
#         return cv2.perspectiveTransform(
#             coors.astype(np.float32), self.mtx_img2obj
#             )

#     def get_imgpoint(self, coors: np.ndarray):
#         '''
#         通过投影，给出物理坐标计算图像坐标

#         ---

#         coors: [[[x,y]],...] or [[[w,h]],...]

#         ---

#         return: [[[x,y]],...]   shape[-1,1,2]
#         '''
#         if len(coors.shape) <= 2:
#             coors = coors.reshape(-1, 1, coors.shape[-1])
#         return cv2.perspectiveTransform(
#             coors.astype(np.float32), self.mtx_obj2img
#             )

#     def get_objpoint_bbox(self, xyxys: np.ndarray):
#         '''
#         通过反投影，给出检测框直接计算物理坐标

#         ---

#         xyxys: [[x1,y1,x2,y2],...] or [[w1,h1,w2,h2],...]

#         ---

#         return: [[[x,y]],...]   shape[-1,1,2]
#         '''
#         if len(xyxys.shape) == 1:
#             xyxys = xyxys.reshape(-1, xyxys.shape[0])
#         coors = np.dstack([
#             np.mean(xyxys[:, [0, 2]], axis=1, keepdims=True),
#             xyxys[:, [3]]
#             ])
#         # coor = np.array([int(np.mean(xyxy[[0,2]])),int(xyxy[3])])
#         return cv2.perspectiveTransform(
#             coors.astype(np.float32), self.mtx_img2obj
#             )

#     def get_distance_absolute(self, pts: np.ndarray):
#         '''
#         给出物理坐标，直接计算欧氏距离

#         ---

#         pts: [[[x,y]],...] shape[-1,1,2]

#         ---

#         return: [d,...]
#         '''
#         if len(pts.shape) <= 2:
#             pts = pts.reshape(-1, 1, pts.shape[-1])
#         return ((pts[:, :, 0]**2 + pts[:, :, 1]**2)**0.5).flatten()

#     def get_distance(self, coor: list, tex_radius=10.0):
#         '''
#         = 不建议使用 =

#         通过图像坐标和贴图半径计算物理距离

#         coor: [y,x] or [h,w]
#         '''
#         v = self.texture_dis[coor[0], coor[1]]
#         dis = self.color_dict[1]['func'](v) * tex_radius
#         return dis

#     def get_distance_bbox(self, xyxy: np.ndarray, tex_radius=10.0):
#         '''
#         = 不建议使用 =

#         通过检测框和贴图半径计算距离

#         xyxy: [x1,y1,x2,y2] or [w1,h1,w2,h2]
#         '''
#         v = self.texture_dis[int(xyxy[3]), int(np.mean(xyxy[[0, 2]]))]
#         dis = self.color_dict[1]['func'](v) * tex_radius
#         return dis

#     def get_distance_combine(self, dpt: np.ndarray, xyxys: np.ndarray):
#         '''
#         由于深度图对x(w)方向预测不佳， 但y(h)还可以，
#         因此结合二者，即使用检测框和投影得到的x和深度图回归的y
#         ---
#         dpt: 深度图
#         xyxys: [[x1,y1,x2,y2]...]

#         ---
#         return: [d,...]
#         '''
#         coors_from_persp = self.get_objpoint_bbox(xyxys)
#         coors_from_depth = self.get_objpoint_from_depth(
#             self.get_depth_bbox(dpt, xyxys)
#             )
#         # 使用反投影结果的x和深度的y
#         cors_combine = np.dstack([
#             coors_from_persp[:, :, 0], coors_from_depth[:, :, -1]
#             ])
#         return self.get_distance_absolute(cors_combine)

# class CameraGUI():

#     def __init__(
#             self,
#             camera: CameraTools3D,
#             camera_id: int = 1,
#             win_name='demo',
#             frame_h=1920,
#             frame_w=1080,
#             win_scale=0.5
#         ) -> None:
#         self.camera = camera
#         self.win_name = win_name
#         self.cap: cv2.VideoCapture = None
#         self.win_h: int = None
#         self.win_w: int = None
#         self.frame_h = frame_h
#         self.frame_w = frame_w
#         self.saved_data: np.ndarray = None

#         self._init_capture(camera_id, win_scale)
#         pass

#     def _init_capture(self, camera_id: int, win_scale: float):
#         cap = cv2.VideoCapture(camera_id)
#         if cap.isOpened():
#             cap_name = cap.getBackendName()
#             print(cap_name)
#         else:
#             raise RuntimeError('不存在的摄像头')
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_w)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_h)
#         cap_prop_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#         cap_prop_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#         print(cap_prop_w, cap_prop_h)
#         self.win_h = int(cap_prop_h * win_scale)
#         self.win_w = int(cap_prop_w * win_scale)
#         self.cap = cap

#     @staticmethod
#     def on_mouse(event, x, y, flags, param: dict):
#         if event == cv2.EVENT_LBUTTONUP:
#             coor = [x, y]
#             if len(param['record_coors']) > 0:
#                 param['record_coors'].append(coor)
#             else:
#                 param['is_drawing'] = True
#                 param['record_coors'] = [coor]
#         if event == cv2.EVENT_MOUSEMOVE:
#             coor = [x, y]
#             param['cur_coor'] = coor

#     def run_get_transform(self):
#         '''
#         一个用于标记透视换算点的图形界面

#         HotKeys
#         ---
#         Q: 退出
#         R: 重置
#         S: 存储
#         ESC: 退出并关闭相机
#         ENTER: 保存并重置
#         '''
#         session_data = {}
#         session_data['cur_coor'] = [0, 0]
#         session_data['record_coors'] = []
#         session_data['is_drawing'] = False
#         saved_data = []
#         cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
#         cv2.resizeWindow(self.win_name, self.win_w, self.win_h)
#         cv2.moveWindow(self.win_name, x=100, y=100)
#         cv2.setMouseCallback(self.win_name, self.on_mouse, session_data)
#         while True:
#             ret, frame = self.cap.read()
#             if not ret:
#                 break
#             frame = self.camera.get_undistort_img(frame)
#             if (len(session_data['record_coors']) > 0
#                     and session_data['is_drawing']):
#                 cv2.polylines(
#                     frame,
#                     pts=[
#                         np.array(
#                             session_data['record_coors']
#                             + [session_data['cur_coor']]
#                             ).astype(np.int32)
#                         ],
#                     isClosed=True,
#                     color=(240, 240, 240),
#                     thickness=2,
#                     )
#             cv2.imshow(self.win_name, frame)
#             key = cv2.waitKey(1000 // 50)
#             if 27 == key:
#                 self.cap.release()
#                 break
#             elif ord('q') == key:
#                 break
#             elif ord('r') == key:
#                 session_data['record_coors'] = []
#             elif ord('s') == key:
#                 saved_data = session_data['record_coors']
#             elif 13 == key:  # enter
#                 saved_data = session_data['record_coors']
#                 session_data['record_coors'] = []
#                 session_data['is_drawing'] = False
#             print(f'\r{saved_data} {session_data}\t\t\t', end='')
#         cv2.destroyWindow(self.win_name)
#         self.saved_data = np.array(saved_data).astype(np.int32)
#         return self.saved_data

#     def run_realtime_detect(self):
#         cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
#         cv2.resizeWindow(self.win_name, self.win_w, self.win_h)
#         cv2.moveWindow(self.win_name, x=100, y=100)
#         stamps = {}
#         while True:
#             stamps['base'] = time.time()
#             ret, frame = self.cap.read()
#             if not ret:
#                 break
#             stamps['read_frame'] = time.time()
#             dst = self.camera.get_undistort_img(frame)
#             stamps['undistort'] = time.time()
#             outs, *_ = detect(dst)
#             outs = outs.astype(np.int32)
#             stamps['dectect'] = time.time()
#             for out in outs:
#                 cv2.rectangle(
#                     dst,
#                     out[0:2],
#                     out[2:4],
#                     color=(255, 255, 255),
#                     thickness=2,
#                     lineType=cv2.LINE_AA,
#                     )
#                 cv2.putText(
#                     dst,
#                     f'{self.camera.get_distance_absolute(self.camera.get_objpoint_bbox(out))/1000}\r\n{self.camera.get_objpoint_bbox(out)[0,0,:]/1000}',
#                     out[[0, 3]],
#                     fontScale=1,
#                     fontFace=cv2.FONT_HERSHEY_DUPLEX,
#                     color=(255, 255, 255),
#                     thickness=2,
#                     lineType=cv2.LINE_AA,
#                     )
#             stamps['draw'] = time.time()
#             cv2.imshow(self.win_name, dst)
#             stamps['imshow'] = time.time()
#             key = cv2.waitKey(1)
#             if 27 == key:  # ESC
#                 self.cap.release()
#                 break
#             if ord('q') == key:
#                 break
#             stamps['wait_key'] = time.time()
#             interval = {
#                 'read_frame': stamps['read_frame'] - stamps['base'],
#                 'undistort': stamps['undistort'] - stamps['read_frame'],
#                 'dectect': stamps['dectect'] - stamps['undistort'],
#                 'draw': stamps['draw'] - stamps['dectect'],
#                 'imshow': stamps['imshow'] - stamps['draw'],
#                 'wait_key': stamps['wait_key'] - stamps['imshow'],
#                 }
#             for k in interval.keys():
#                 interval[k] = round(interval[k] * 1000, 2)
#             stamps['calc'] = time.time()
#             print(
#                 f"\rfps: {1/(stamps['calc']-stamps['base']):>4.1f} {interval}",
#                 end=''
#                 )
#         cv2.destroyWindow(self.win_name)

#     def run_realtime_track(self, **kwds):
#         '''
#         thr_1: 级联匹配的阈值(投影面的欧氏距离)
#         thr_2: IOU匹配的阈值
#         '''
#         cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
#         cv2.resizeWindow(self.win_name, self.win_w, self.win_h)
#         cv2.moveWindow(self.win_name, x=100, y=100)
#         _, frame = self.cap.read()
#         frame = self.camera.get_undistort_img(frame)
#         outs, *_ = detect(frame, conf=0.3, iou=0.3)
#         TM = TrackMaster(
#             outs,
#             **kwds,
#             mtx_img2obj=self.camera.mtx_img2obj,
#             )
#         while True:
#             ret, frame = self.cap.read()
#             if not ret:
#                 break
#             frame = self.camera.get_undistort_img(frame)
#             # outs, *_, kpts = pose_detect(frame, conf=0.3, iou=0.3)
#             outs, *_, kpts = pose_predictor(
#                 frame,
#                 conf=0.3,
#                 iou=0.3,
#                 image_shape=(self.frame_h, self.frame_w),
#                 )
#             tic = time.time()
#             TM.update(outs, is_delete=True)
#             toc = time.time()
#             print(f'\rfps: {1/(toc-tic+0.000001)}', end='')
#             # disp(TM.tracks)
#             frame = draw_box(frame, outs.astype(np.int32))
#             frame = draw_text(
#                 frame,
#                 arr_text=[
#                     f'{tid}' for tid in TM.get_latest_record(
#                         TM.get([
#                             TM.STATE_CONFIRMED, TM.STATE_TENTATIVE,
#                             TM.STATE_SIGNALLES
#                             ])
#                         ).keys()
#                     ],
#                 outs=TM.get_latest_record(
#                     TM.get([
#                         TM.STATE_CONFIRMED,
#                         TM.STATE_TENTATIVE,
#                         TM.STATE_SIGNALLES
#                         ])
#                     ).values(),
#                 color=(255, 255, 0),
#                 fontScale=8,
#                 thickness=4
#                 )
#             try:
#                 frame = draw_text(
#                     frame,
#                     # arr_text=[
#                     #     f'{self.camera.get_distance_absolute(self.camera.get_objpoint_bbox(out))/1000}\r\n{self.camera.get_objpoint_bbox(out)[0,0,:]/1000}'
#                     #     for out in outs
#                     #     ],
#                     arr_text=[
#                         f'=={self.camera.get_distance_absolute(self.camera.get_objpoint(kpt[-1]))/1000}'
#                         for kpt in kpts
#                         ],
#                     outs=outs,
#                     color=(255, 255, 255),
#                     fontScale=3,
#                     thickness=2
#                     )
#                 for kpt in kpts:
#                     frame = draw_pts(
#                         frame, kpt.astype(np.int32), color=(122, 240, 65)
#                         )
#             except:
#                 ...
#             cv2.imshow(self.win_name, frame)
#             key = cv2.waitKey(5)
#             if ord('q') == key:
#                 break
#             if 27 == key:
#                 self.cap.release()
#                 break
#         cv2.destroyWindow(self.win_name)
