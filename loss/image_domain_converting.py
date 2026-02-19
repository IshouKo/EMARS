## 画像の色空間（RGBとリニア空間）の変換を行う関数
import numpy as np
import torch
from absl.logging import info


"""
・入力: 0〜1の範囲のRGB画像（NumPy配列またはPyTorchテンソル）
ガンマ補正を使ってRGB値をリニア空間に変換します（通常はガンマ2.2）。
値が0〜1の範囲外なら警告を出し
"""
def rgb_to_lineal(matrix, gamma=2.2):
    """
    :param matrix: rgb matrix, value range is[0, 1]
    :param gamma:
    :return: linear matrix, value range is [0, 1]
    """
    if matrix.max() > 1 or matrix.min() < 0:
        info(f"rgb_to_lineal: value range not is [0, 1], ({matrix.min()}, {matrix.max()})")
    if isinstance(matrix, torch.Tensor):
        return torch.pow(matrix, gamma)
    return np.power(matrix, gamma)


def lineal_to_rgb(matrix, gamma=2.2):
    """
    :param matrix: linear matrix, value range is [0, 1]
    :param gamma:
    :return: rgb matrix, value range is[0, 1]
    """
    if matrix.max() > 1 or matrix.min() < 0:
        info(f"lineal_to_rgb: value range not is [0, 1], ({matrix.min()}, {matrix.max()})")
    if isinstance(matrix, torch.Tensor):
        return torch.pow(matrix, 1 / gamma)
    return np.power(matrix, 1 / gamma)
