
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from qfluentwidgets import *
from .sti_rect import StiRect
import threading
import time
import numpy as np
from .dataPLot_window import DataAnalysisPlotModule
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from scipy.io import savemat
import os
class StimTestWindow(QWidget):
    def __init__(self, objectName):
        super().__init__()
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)
        self.setObjectName(objectName)
        self.setQss()

        self.cache_data = []
        self.finish = True

        self.__layout__()
        self.__items__()

    def setQss(self):
        with open(f'source/qss/mainWindow.qss', encoding='utf-8') as f:
            self.setStyleSheet(f.read())

    def __layout__(self):
        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.setLayout(self.layout)

    def __items__(self):
        self.stim_area = QWidget(objectName='stim_test_area')
        self.stim_layout = QVBoxLayout(self.stim_area)
        self.stim_layout.setContentsMargins(5, 5, 5, 5)
        self.stim_layout.setSpacing(5)
        self.layout.addWidget(self.stim_area)

        self.rect_10 = StiRect('10Hz', self)
        self.rect_10.setFixedSize(200, 200)
        self.stim_layout.addWidget(self.rect_10, alignment=Qt.AlignCenter)

        self.dataAnalysis_area = QWidget()
        self.dataAnalysis_layout = QHBoxLayout(self.dataAnalysis_area)
        self.dataAnalysis_layout.setContentsMargins(0, 5, 5, 5)
        self.dataAnalysis_layout.setSpacing(5)
        self.layout.addWidget(self.dataAnalysis_area)

        self.dataAnalysis_layout_1 = QVBoxLayout(self.dataAnalysis_area)
        self.dataAnalysis_layout_1.setContentsMargins(0, 0, 0, 0)
        self.dataAnalysis_layout_1.setSpacing(5)
        self.dataAnalysis_layout.addLayout(self.dataAnalysis_layout_1)

        self.dataAnalysis_layout_2 = QVBoxLayout(self.dataAnalysis_area)
        self.dataAnalysis_layout_2.setContentsMargins(0, 0, 0, 0)
        self.dataAnalysis_layout_2.setSpacing(5)
        self.dataAnalysis_layout.addLayout(self.dataAnalysis_layout_2)

        self.image_1 = DataAnalysisPlotModule('1')
        self.image_2 = DataAnalysisPlotModule('2')
        self.image_3 = DataAnalysisPlotModule('3')
        self.image_4 = DataAnalysisPlotModule('4')
        self.image_5 = DataAnalysisPlotModule('5')
        self.image_6 = DataAnalysisPlotModule('6')
        self.image_7 = DataAnalysisPlotModule('7')
        self.image_8 = DataAnalysisPlotModule('8')

        # self.image_1 = ImageLabel()
        # self.image_2 = ImageLabel()
        # self.image_3 = ImageLabel()
        # self.image_4 = ImageLabel()
        # self.image_5 = ImageLabel()
        # self.image_6 = ImageLabel()
        # self.image_7 = ImageLabel()
        # self.image_8 = ImageLabel()

        self.dataAnalysis_layout_1.addWidget(self.image_1)
        self.dataAnalysis_layout_1.addWidget(self.image_2)
        self.dataAnalysis_layout_1.addWidget(self.image_3)
        self.dataAnalysis_layout_1.addWidget(self.image_4)
        self.dataAnalysis_layout_2.addWidget(self.image_5)
        self.dataAnalysis_layout_2.addWidget(self.image_6)
        self.dataAnalysis_layout_2.addWidget(self.image_7)
        self.dataAnalysis_layout_2.addWidget(self.image_8)

        for i in range(1, 9):
            image = getattr(self, f'image_{i}')
            image.setFixedHeight(200)
            image.setObjectName('fft_image')

    def resizeEvent(self, event):
        super().resizeEvent(event)
        width = self.width()
        height = self.height()
        self.stim_area.setFixedWidth(width * 0.3)

    def mousePressEvent(self, a0):
        super().mousePressEvent(a0)
        if a0.button() == Qt.RightButton:
            self.start_sti_event()

    def setDefaultColor(self):
        self.rect_10.setDefaultColor()
        QApplication.processEvents()

    def start_sti_event(self):
        if self.finish:
            self.finish = False
            th = threading.Thread(target=self.flick)
            th.start()
        else:
            self.finish = True
            self.setDefaultColor()

    def flick(self):
        fps = 240
        ifi = 1 / fps
        frameNum = 0
        phase = 1
        start_time = time.time()

        while not self.finish:
            self.rect_10.changeColor(10, frameNum, phase * np.pi, ifi)

            end_time = time.time()
            frameNum = (end_time - start_time) * fps
            time.sleep(0.000001)

    def getData(self, data):
        if not self.finish:
            self.cache_data.append(data)

            if len(self.cache_data) == 12:
                self.showFFTResultThread(self.cache_data)
                data = np.concatenate(self.cache_data, axis=-1)
                savemat('test.mat', {'data': data})
                self.cache_data = []

    def showFFTResultThread(self, data):
        th = threading.Thread(target=self.showFFTResult, args=([data]))
        th.start()

    def showFFTResult(self, data):
        data = np.concatenate(data, axis=-1)

        Fs = 250
        T = 1 / Fs
        L = data.shape[-1]
        t = np.arange(0, L) * T

        N = 65536
        freq = np.linspace(0, Fs, L)
        width = self.width()
        for i in range(data.shape[0]):
            channcel_data = data[i]
            # Y = np.fft.fft(channcel_data, N) / N * 2
            # A = abs(Y)
            # A = A[ : N // 2]

            y = np.fft.fft(channcel_data)
            y_amp = abs(y)

            # plt.figure()
            # plt.plot(freq, y_amp)
            # plt.savefig(f'{i}.jpg')

            # plt.show()

            image = getattr(self, f'image_{i + 1}')
            image.clearData()
            image.set_data(y_amp)
            # time.sleep(0.00001)
            # image.setImage(f'{i}.jpg')
            # image.setFixedSize(width * 0.3, 110)
            QApplication.processEvents()
