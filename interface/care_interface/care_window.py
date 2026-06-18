import os
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
from .sti_rect import StiRect
from scipy import signal
from scipy.io import savemat
from glob import glob
import time
from models.TDCA import TDCA
from config import config

class CareControlWindow(QWidget):
    def __init__(self, objectName):
        super().__init__()
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)
        self.setObjectName(objectName)
        self.setQss()

        self.sti_lst = [
            8, 8.5, 9, 9.5, 15.6, 10.5, 11, 11.5, 12
        ]
        self.cache_data = np.array([])

        self.rect_size = 150
        self.start_flick = False
        self.finish = True
        self.start_cache = False
        self._initLayout()
        self._initItems()

        self.times = 1
        self.fbcca = TDCA(3, self.times, self.sti_lst)
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
        self.layout.addSpacing(80)

        self.sti_rects = []

        self.hLayout1 = QHBoxLayout()
        self.hLayout1.setContentsMargins(5, 5, 5, 5)
        self.hLayout1.setSpacing(0)
        self.layout.addLayout(self.hLayout1)

        self.eat_rect = StiRect('source/icons/eat.png', self, self.sti_lst[0], '我想吃饭')
        self.eat_rect.setFixedSize(self.rect_size, self.rect_size)
        self.hLayout1.addWidget(self.eat_rect)
        self.sti_rects.append(self.eat_rect)

        self.up_rect = StiRect('source/icons/up.png', self, self.sti_lst[1], '把床升起来')
        self.up_rect.setFixedSize(self.rect_size, self.rect_size)
        self.hLayout1.addWidget(self.up_rect)
        self.sti_rects.append(self.up_rect)

        self.water_rect = StiRect('source/icons/water.png', self, self.sti_lst[2], '我想喝水')
        self.water_rect.setFixedSize(self.rect_size, self.rect_size)
        self.hLayout1.addWidget(self.water_rect)
        self.sti_rects.append(self.water_rect)

        ##################################
        self.layout.addSpacing(120)

        self.hLayout2 = QHBoxLayout()
        self.hLayout2.setContentsMargins(5, 5, 5, 5)
        self.hLayout2.setSpacing(0)
        self.layout.addLayout(self.hLayout2)

        self.music_rect = StiRect('source/icons/music.png', self, self.sti_lst[3], '我想听音乐')
        self.music_rect.setFixedSize(self.rect_size, self.rect_size)
        self.hLayout2.addWidget(self.music_rect)
        self.sti_rects.append(self.music_rect)

        self.emergent_rect = StiRect('source/icons/emergent.png', self, self.sti_lst[4], '紧急情况')
        self.emergent_rect.setFixedSize(self.rect_size, self.rect_size)
        self.hLayout2.addWidget(self.emergent_rect)
        self.sti_rects.append(self.emergent_rect)

        self.tv_rect = StiRect('source/icons/tv.png', self, self.sti_lst[5], '我想看电视')
        self.tv_rect.setFixedSize(self.rect_size, self.rect_size)
        self.hLayout2.addWidget(self.tv_rect)
        self.sti_rects.append(self.tv_rect)

        ##################################
        self.layout.addSpacing(120)

        self.hLayout3 = QHBoxLayout()
        self.hLayout3.setContentsMargins(5, 5, 5, 5)
        self.hLayout3.setSpacing(0)
        self.layout.addLayout(self.hLayout3)

        self.sleep_rect = StiRect('source/icons/sleep.png', self, self.sti_lst[6], '我想睡觉')
        self.sleep_rect.setFixedSize(self.rect_size, self.rect_size)
        self.hLayout3.addWidget(self.sleep_rect)
        self.sti_rects.append(self.sleep_rect)

        self.down_rect = StiRect('source/icons/down.png', self, self.sti_lst[7], '把床降下去')
        self.down_rect.setFixedSize(self.rect_size, self.rect_size)
        self.hLayout3.addWidget(self.down_rect)
        self.sti_rects.append(self.down_rect)

        self.wc_rect = StiRect('source/icons/wc.png', self, self.sti_lst[8], '我想上厕所')
        self.wc_rect.setFixedSize(self.rect_size, self.rect_size)
        self.hLayout3.addWidget(self.wc_rect)
        self.sti_rects.append(self.wc_rect)
        self.layout.addSpacerItem(QSpacerItem(1, 1, QSizePolicy.Maximum, QSizePolicy.Expanding))

    def setDefaultColor(self):
        for rect in self.sti_rects:
            rect.setDefaultColor()

    def start_sti_event(self):
        if not self.start_flick:
            self.start_flick = True
            self.finish = False
            self.start_cache = False
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
                    rect.changeColor(rect.sti, end_time - start_time)

                self.progress.setValue((self.cache_data.shape[-1] / (self.times * 250)) * 100)
                time.sleep(0.00001)

            self.progress.setValue(100)
            self.setDefaultColor()
            self.start_cache = False
            if not self.finish and len(self.cache_data) != 0:
                used_data = self.cache_data[:, -self.times * 250:]

                savePath = os.path.join('saveCareData', config.subjectName)
                if not os.path.exists(savePath):
                    os.makedirs(savePath)

                fileNums = len(glob(os.path.join(savePath, '*.mat')))
                saveFile = os.path.join(savePath, f'{fileNums + 1}.mat')
                savemat(saveFile, {'data': used_data})

                self.cache_data = np.array([])
                result = self.fbcca.classify(used_data)
                result = int(result)
                self.set_result(result)
            time.sleep(0.01)

    def set_result(self, idx):
        rect = self.sti_rects[idx]
        pyttsx3.speak(rect.text)
        self.show_label.setText(rect.text)

    def getData(self, data):
        if self.finish:
            return

        if self.start_cache:
            if len(self.cache_data) == 0:
                self.cache_data = data
            else:
                self.cache_data = np.concatenate([self.cache_data, data], axis=-1)