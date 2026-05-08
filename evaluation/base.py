import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
import cv2
from scipy.optimize import linear_sum_assignment

# ===== EVALUATION ===== #


def batch_iou(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    bboxes1 : array
        行
    bboxes2 : array
        列

    Returns
    -------
    iou : np.ndarray
        以 bboxes1 为行，bboxes2 为列的 IoU 矩阵
    """
    C = np.hstack([
        np.repeat(bboxes1, bboxes2.shape[0], axis=0),
        np.tile(bboxes2, (bboxes1.shape[0], 1)),
        ]).reshape([-1, 2, 4])
    D = np.stack([
        np.max(C[:, :, [0, 1]], axis=1),
        np.min(C[:, :, [2, 3]], axis=1),
        ],
                    axis=1)
    D = (D[:, 1, :] - D[:, 0, :]).clip(0)
    I = D[:, 0] * D[:, 1]
    U1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    U2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    iou: np.ndarray = I / (
        np.repeat(U1, bboxes2.shape[0]) + np.tile(U2, bboxes1.shape[0]) - I
        )
    iou = iou.reshape(bboxes1.shape[0], bboxes2.shape[0])

    return iou


def get_match_arg(cost_mat: np.ndarray, threshold: float = 0.1):
    """
    获取大于阈值的匈牙利法匹配对

    Parameters
    ----------
    cost_mat : np.ndarray
        代价矩阵
    threshold : float
        阈值
    
    Returns
    -------
    b1 : np.ndarray
        匹配的行
    b2 : np.ndarray
        匹配的列
    """
    b1, b2 = linear_sum_assignment(-cost_mat)
    list_match = cost_mat[b1, b2]
    arg_TP = np.where(list_match >= threshold)
    return b1[arg_TP], b2[arg_TP]


def get_evaluation(gt: np.ndarray, pred: np.ndarray):
    """
    获取 precision, recall, f1score 指标

    Parameters
    ----------
    gt : np.ndarray
        真实值 [x1, y1, x2, y2]
    pred : np.ndarray
        预测值 [x1, y1, x2, y2]
    
    Returns
    -------
    precision : float
        精确率
    recall : float
        召回率
    f1score : float
        F1 分数
    """
    cost_mat = batch_iou(gt, pred)
    b1, b2 = get_match_arg(cost_mat, 0.01)
    precision = len(b1) / (len(pred))
    recall = len(b2) / (len(gt))
    if precision * recall == 0:
        f1score = 0.0
    else:
        f1score = 2 * (precision*recall) / (precision+recall)
    return precision, recall, f1score


def get_evaluation_multicls(
        gt: np.ndarray,
        gt_lbls: np.ndarray,
        pred: np.ndarray,
        pred_lbls: np.ndarray,
        offset: int = None
    ):
    """
    获取多类别 precision, recall, f1score 指标
    计算方式为 micro-f1;

    Parameters
    ----------
    gt : np.ndarray
        真实值 [[x1, y1, x2, y2], ...]
    gt_lbls : np.ndarray
        真实值标签 [0 1 2 3 5 7, ...]
    pred : np.ndarray
        预测值 [[x1, y1, x2, y2], ...]
    pred_lbls : np.ndarray
        预测值标签 [0 1 2 3 5 7, ...]
    offset : int
        偏移量，手动指定即图像的最大边像素长度；不指定则默认为输入坐标的最大尺寸
    """
    if None is offset:
        offset = max(gt.max(), pred.max()) + 100
    cost_mat = batch_iou(
        gt + gt_lbls[:, None] * offset, pred + pred_lbls[:, None] * offset
        )
    b1, b2 = get_match_arg(cost_mat, 0.01)
    precision = len(b1) / (len(pred))
    recall = len(b2) / (len(gt))
    if precision * recall == 0:
        f1score = 0.0
    else:
        f1score = 2 * (precision*recall) / (precision+recall)
    return precision, recall, f1score
