import socket
import pyttsx3
from Pinyin2Hanzi import DefaultDagParams
from Pinyin2Hanzi import dag
import matplotlib.pyplot as plt
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from qfluentwidgets import *
from qframelesswindow import *
from qfluentwidgets import FluentIcon as FIF
import threading
import numpy as np
from .sti_rect import DoubleStiRect
from scipy import signal
from scipy.io import savemat
from glob import glob
import time
from models.TDCA import TDCA

class DualFrequencyWindow(QWidget):
    def __init__(self, objectName):
        super().__init__()
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)
        self.setObjectName(objectName)
        self.setQss()

        self.sti_left = [15 + 0.5 * i for i in range(10)]
        self.sti_right = [8 + 0.5 * i for i in range(12)]
        # self.sti_right = [
        #     35.6, 30.8, 36.4, 31.6, 37.2, 32.4, 38, 33.2, 38.8, 34, 39.6, 34.8
        # ]
        self.cache_data = np.array([])

        self.rect_size = 80
        self.start_flick = False
        self.finish = True
        self.start_cache = False
        self._initLayout()
        self._initItems()

        self.times = 1
        self.fbcca_left = TDCA(3, self.times, self.sti_left, Nh=3)
        self.fbcca_right = TDCA(3, self.times, self.sti_right, Nh=3)

        # self.fbcca.sendResultSignal.connect(self.set_result)

    def setQss(self):
        with open(f'source/qss/mainWindow.qss', encoding='utf-8') as f:
            self.setStyleSheet(f.read())

    def _initLayout(self):
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(1)
        self.setLayout(self.layout)

    def _initItems(self):
        self.show_label = DisplayLabel()
        self.show_label.setFixedHeight(65)
        self.layout.addWidget(self.show_label)

        self.progress = ProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.layout.addWidget(self.progress)
        self.layout.addSpacing(10)

        self.sti_rects = []

        self.flowLayout = FlowLayout()
        self.flowLayout.setVerticalSpacing(35)
        self.flowLayout.setHorizontalSpacing(35)
        self.flowLayout.setContentsMargins(50, 5, 10, 0)
        self.layout.addLayout(self.flowLayout)

        # vSpacing = self.flowLayout.verticalSpacing()
        # hSpacing = self.flowLayout.horizontalSpacing()
        # contentsMargins = self.flowLayout.contentsMargins()
        #
        # cardWidth = (self.width() - contentsMargins.left() - contentsMargins.right()) // 10
        # cardWidth = cardWidth - hSpacing
        # cardHeight = cardWidth

        cardWidth = 115
        cardHeight = cardWidth
        self.sti_char_dict = {}
        for i in range(60):
            col = i % 10
            row = i // 5
            left_sti = self.sti_left[col]
            right_sti = self.sti_right[row]

            char = f'{i+1}'
            rect = DoubleStiRect(char, left_sti, right_sti)
            rect.setFixedSize(cardWidth, cardHeight)
            self.flowLayout.addWidget(rect)
            self.sti_rects.append(rect)

            self.sti_char_dict[f'{col}{row}'] = char

    def setDefaultColor(self):
        for rect in self.sti_rects:
            rect.setDefaultColor()

    def start_sti_event(self):
        if not self.start_flick:
            self.start_flick = True
            self.finish = False
            self.cache_data = np.array([])
            self.show_label.setText('')
            th = threading.Thread(target=self.flick)
            th.start()

        else:
            self.start_flick = False
            self.finish = True
            self.start_cache = False
            self.setDefaultColor()
            self.show_label.setText('')

    def flick(self):
        while not self.finish:
            start_time = time.time()
            self.cache_data = np.array([])
            self.start_cache = True
            while self.cache_data.shape[-1] < self.times * 250 and not self.finish:
                end_time = time.time()
                for idx, rect in enumerate(self.sti_rects):
                    rect.changeColor(end_time - start_time)

                self.progress.setValue((self.cache_data.shape[-1] / (self.times * 250)) * 100)
                time.sleep(0.00001)

            self.progress.setValue(100)
            self.setDefaultColor()
            self.start_cache = False
            if not self.finish and len(self.cache_data) != 0:
                used_data = self.cache_data[:, -self.times * 250:]
                self.cache_data = np.array([])

                left_data = used_data[[3, 5, 6], :]
                right_data = used_data[[0, 1, 7], :]

                result_left = self.fbcca_left.classify(right_data)
                result_right = self.fbcca_right.classify(left_data)

                result_left = int(result_left)
                result_right = int(result_right)

                sti = f'{result_left}{result_right}'
                if sti not in self.sti_char_dict:
                    result = '-1'
                else:
                    result = self.sti_char_dict[f'{result_left}{result_right}']
                self.set_result(result)
            time.sleep(0.01)

    def set_result(self, res):
        pyttsx3.speak(res)
        self.show_label.setText(res)

    def getData(self, data):
        if self.finish:
            return

        if self.start_cache:
            if len(self.cache_data) == 0:
                self.cache_data = data
            else:
                self.cache_data = np.concatenate([self.cache_data, data], axis=-1)