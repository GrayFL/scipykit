from scipykit.utils import *
import numpy as np


# region Skeleton
class Skeleton():
    '''
    Skeleton不提供归一化能力，请在使用Sk转换前将数据集尺度统一
    '''

    def __init__(self) -> None:
        ...

    @staticmethod
    def __pts2vecs(
            src: np.ndarray,
            datatype: str,
            mode='basic',
            image_shape: tuple[int, int] = None
        ):
        '''
        Parameters
        ---
        mode:
            - `basic`: [肩中, 髋中, *骨骼(量化)]
        '''
        if 'football' == datatype:
            body_pts = {
                '头中': np.mean(src[[12, 13], :], axis=0),
                '肩中': np.mean(src[[8, 9], :], axis=0),
                '髋中': np.mean(src[[2, 3], :], axis=0),
                '左肩': src[9],
                '右肩': src[8],
                '左肘': src[10],
                '右肘': src[7],
                '左手': src[11],
                '右手': src[6],
                '左髋': src[2],
                '右髋': src[3],
                '左膝': src[1],
                '右膝': src[4],
                '左脚': src[0],
                '右脚': src[5],
                }
            body_bones = {
                '躯干骨': body_pts['肩中'] - body_pts['髋中'],
                '头颈骨': body_pts['头中'] - body_pts['肩中'],
                '左肩胛': body_pts['左肩'] - body_pts['肩中'],
                '右肩胛': body_pts['右肩'] - body_pts['肩中'],
                '左大臂': body_pts['左肘'] - body_pts['左肩'],
                '右大臂': body_pts['右肘'] - body_pts['右肩'],
                '左小臂': body_pts['左手'] - body_pts['左肘'],
                '右小臂': body_pts['右手'] - body_pts['右肘'],
                '左髋骨': body_pts['左髋'] - body_pts['髋中'],
                '右髋骨': body_pts['右髋'] - body_pts['髋中'],
                '左大腿': body_pts['左膝'] - body_pts['左髋'],
                '右大腿': body_pts['右膝'] - body_pts['右髋'],
                '左小腿': body_pts['左脚'] - body_pts['左膝'],
                '右小腿': body_pts['右脚'] - body_pts['右膝'],
                }
            bone_main_len = np.linalg.norm(body_bones['躯干骨'])
        elif 'coco-wholebody' == datatype:
            src = src[:, :2]
            body_pts = {
                '头中': src[0],
                '肩中': np.mean(src[[5, 6], :], axis=0),
                '髋中': np.mean(src[[11, 12], :], axis=0),
                '左肩': src[5],
                '右肩': src[6],
                '左肘': src[7],
                '右肘': src[8],
                '左手': src[9],
                '右手': src[10],
                '左髋': src[11],
                '右髋': src[12],
                '左膝': src[13],
                '右膝': src[14],
                '左脚': src[15],
                '右脚': src[16],
                }
            body_bones = {
                '躯干骨': body_pts['肩中'] - body_pts['髋中'],
                '头颈骨': body_pts['头中'] - body_pts['肩中'],
                '左肩胛': body_pts['左肩'] - body_pts['肩中'],
                '右肩胛': body_pts['右肩'] - body_pts['肩中'],
                '左大臂': body_pts['左肘'] - body_pts['左肩'],
                '右大臂': body_pts['右肘'] - body_pts['右肩'],
                '左小臂': body_pts['左手'] - body_pts['左肘'],
                '右小臂': body_pts['右手'] - body_pts['右肘'],
                '左髋骨': body_pts['左髋'] - body_pts['髋中'],
                '右髋骨': body_pts['右髋'] - body_pts['髋中'],
                '左大腿': body_pts['左膝'] - body_pts['左髋'],
                '右大腿': body_pts['右膝'] - body_pts['右髋'],
                '左小腿': body_pts['左脚'] - body_pts['左膝'],
                '右小腿': body_pts['右脚'] - body_pts['右膝'],
                }
            bone_main_len = np.linalg.norm(body_bones['躯干骨'])
        elif 'mmpose3d' == datatype:
            body_pts = {
                '头中': np.mean(src[[8, 10], :], axis=0),
                '肩中': np.mean(src[[11, 14], :], axis=0),
                '髋中': src[0],
                '左肩': src[11],
                '右肩': src[14],
                '左肘': src[12],
                '右肘': src[15],
                '左手': src[11],
                '右手': src[16],
                '左髋': src[4],
                '右髋': src[1],
                '左膝': src[5],
                '右膝': src[2],
                '左脚': src[6],
                '右脚': src[3],
                }
            body_bones = {
                '躯干骨': body_pts['肩中'] - body_pts['髋中'],
                '头颈骨': body_pts['头中'] - body_pts['肩中'],
                '左肩胛': body_pts['左肩'] - body_pts['肩中'],
                '右肩胛': body_pts['右肩'] - body_pts['肩中'],
                '左大臂': body_pts['左肘'] - body_pts['左肩'],
                '右大臂': body_pts['右肘'] - body_pts['右肩'],
                '左小臂': body_pts['左手'] - body_pts['左肘'],
                '右小臂': body_pts['右手'] - body_pts['右肘'],
                '左髋骨': body_pts['左髋'] - body_pts['髋中'],
                '右髋骨': body_pts['右髋'] - body_pts['髋中'],
                '左大腿': body_pts['左膝'] - body_pts['左髋'],
                '右大腿': body_pts['右膝'] - body_pts['右髋'],
                '左小腿': body_pts['左脚'] - body_pts['左膝'],
                '右小腿': body_pts['右脚'] - body_pts['右膝'],
                }
            bone_main_len = np.linalg.norm(body_bones['躯干骨'])
        if 'basic' == mode:
            s = np.array(image_shape[::-1])  # 图像大小 [w, h]
            vecs = np.vstack(
                [
                    body_pts['肩中'] / s,  # 在图像中的相对位置
                    body_pts['髋中'] / s,
                    ] +
                [bone / bone_main_len for bone in body_bones.values()],
                dtype=np.float32,
                )
        elif 'relative' == mode:
            vecs = np.vstack(
                [
                    body_pts['肩中'],
                    body_pts['髋中'],
                    ] +
                [bone / bone_main_len for bone in body_bones.values()],
                dtype=np.float32,
                )
        return vecs

    @staticmethod
    def __vecs2pts(
            src: np.ndarray,
            datatype: str,
            mode='basic',
            image_shape: tuple[int, int] = None
        ):
        '''
        Parameters
        ---
        mode:
            - `basic`: [肩中, 髋中, *骨骼(量化)]
        '''
        if 'basic' == mode:
            s = np.array(image_shape[::-1])  # 图像大小 [w, h]
            body_pts = {
                '头中': None,
                '肩中': src[0] * s,
                '髋中': src[1] * s,
                }
            vecs = src[2:32
                       ] * np.linalg.norm(body_pts['肩中'] - body_pts['髋中'])
        if 'relative' == mode:
            body_pts = {
                '头中': None,
                '肩中': src[0],
                '髋中': src[1],
                }
            vecs = src[2:32
                       ] * np.linalg.norm(body_pts['肩中'] - body_pts['髋中'])
        body_bones = {
            '躯干骨': vecs[0],
            '头颈骨': vecs[1],
            '左肩胛': vecs[2],
            '右肩胛': vecs[3],
            '左大臂': vecs[4],
            '右大臂': vecs[5],
            '左小臂': vecs[6],
            '右小臂': vecs[7],
            '左髋骨': vecs[8],
            '右髋骨': vecs[9],
            '左大腿': vecs[10],
            '右大腿': vecs[11],
            '左小腿': vecs[12],
            '右小腿': vecs[13],
            }
        body_pts['头中'] = body_pts['肩中'] + body_bones['头颈骨']
        body_pts['肩中']
        body_pts['髋中']
        body_pts['左肩'] = body_pts['肩中'] + body_bones['左肩胛']
        body_pts['右肩'] = body_pts['肩中'] + body_bones['右肩胛']
        body_pts['左肘'] = body_pts['左肩'] + body_bones['左大臂']
        body_pts['右肘'] = body_pts['右肩'] + body_bones['右大臂']
        body_pts['左手'] = body_pts['左肘'] + body_bones['左小臂']
        body_pts['右手'] = body_pts['右肘'] + body_bones['右小臂']
        body_pts['左髋'] = body_pts['髋中'] + body_bones['左髋骨']
        body_pts['右髋'] = body_pts['髋中'] + body_bones['右髋骨']
        body_pts['左膝'] = body_pts['左髋'] + body_bones['左大腿']
        body_pts['右膝'] = body_pts['右髋'] + body_bones['右大腿']
        body_pts['左脚'] = body_pts['左膝'] + body_bones['左小腿']
        body_pts['右脚'] = body_pts['右膝'] + body_bones['右小腿']
        pts = np.array([pt for pt in body_pts.values()], dtype=np.float32)
        bones = np.array([bone for bone in body_bones.values()],
                            dtype=np.float32)
        return pts, bones

    @staticmethod
    def cvt(
            src: np.ndarray,
            datatype: str,
            mode: str,
            style='basic',
            **kwds
        ) -> np.ndarray:
        '''
        Parameters
        ---
        mode:
            - `pts2flat`: 所有元素展平成列向量的形式 (-1, 1)
            - `pts2vector`: 向量形式 (-1, 2)
            - `pts2pattern`: 向量对，一左一右 (-1, 4)
            - 反向同理，如 `vector2pts`
        image_shape:
            图像大小，全局统一格式 [h, w]
        '''
        if 'pts2flat' == mode:
            vecs = Skeleton.__pts2vecs(src, datatype, style, **kwds)
            return vecs.reshape(-1, 1)
        if 'pts2vector' == mode:
            vecs = Skeleton.__pts2vecs(src, datatype, style, **kwds)
            return vecs
        if 'pts2pattern' == mode:
            vecs = Skeleton.__pts2vecs(src, datatype, style, **kwds)
            return vecs.reshape(-1, 4)
        if 'vector2pts' == mode:
            pts, bones = Skeleton.__vecs2pts(src, datatype, style, **kwds)
            return pts, bones

    # endregion
