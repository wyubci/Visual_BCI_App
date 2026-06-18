import cv2
from PyQt5.QtGui import QImage, QPixmap
import numpy as np
import time


def cvimage2qtpixmap(image):
    """
    将cv2格式的image转换为QPixmap格式
    :param image:
    :return:
    """
    if len(image.shape) == 3:
        height, width, depth = image.shape
    else:
        height, width= image.shape
        depth = 3

    if depth == 4:
        cvimg = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        qimg = QImage(cvimg.data, width, height, width * depth, QImage.Format_RGBA8888)
    else:
        cvimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        qimg = QImage(cvimg.data, width, height, width * depth, QImage.Format_RGB888)

    qpixmap = QPixmap.fromImage(qimg)

    return qpixmap