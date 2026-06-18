import random
import time
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

# from ..user_interface.userCreate_window import UserCreater
from config import config

# from controller.recognition_controller import recognition
from models.FBCCA import FBCCA

class KeyboardControlWindow(QWidget):
    startEEGRecording = pyqtSignal(object)
    endEEGRecording = pyqtSignal()
    def __init__(self, objectName):
        super().__init__()
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)
        self.setObjectName(objectName)
        self.setQss()

        self.readty_sti_list = []
        self.result_sti_list = []
        # self.sti_lst = [8 + i * 0.2 for i in range(36)]
        self.sti_lst = config.sti_lst

        self.char_list = [
            ['back', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M'],
            ['Space']
        ]
        self.char_list_all = [
            'back', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
            'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P',
            'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L',
            'Z', 'X', 'C', 'V', 'B', 'N', 'M',
            'Space'
        ]

        self.show_chars = []
        self.hide_index = []
        self.cache_data = []
        self.candidate_chars = []
        self.start_flick = False
        self.finish = True
        self.collect_flag = False
        self.currentItem = None
        self.result_times = 0
        self.startCollect = False
        self.currentCollectSti = None
        self.start_cache = False
        self._initLayout()
        self._initItems()

        config.currentUser = None
        config.save()

        # recognition.trainingProcessSignal.connect(self.setTrainingProcessEvent)

        self.dagParams = DefaultDagParams()

        self.times = 4
        self.fbcca = FBCCA(3, self.times, self.sti_lst)
        self.fbcca.sendResultSignal.connect(self.set_result)

        # self.candidateFBCCA = FBCCA(3, config.candidateStiList)


    def setQss(self):
        with open(f'source/qss/mainWindow.qss', encoding='utf-8') as f:
            self.setStyleSheet(f.read())

    def resizeEvent(self, a0):
        super().resizeEvent(a0)
        # self.stimulate_area.setFixedSize(self.width(), self.height() * 0.8)

    def _initLayout(self):
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(1)
        self.setLayout(self.layout)

    def _initItems(self):
        sti_idx = 1
        self.sti_rects = {}

        h1 = QHBoxLayout()
        h1.setContentsMargins(0, 0, 0, 0)
        h1.setSpacing(10)
        self.layout.addLayout(h1)

        font = QFont()
        font.setPixelSize(60)
        self.output_widget = QWidget()
        self.output_widget.setFixedSize(1270, 130)
        self.output_layout = QVBoxLayout(self.output_widget)
        self.output_layout.setContentsMargins(0, 0, 0, 0)
        self.output_layout.setSpacing(5)

        self.output_area = LineEdit()
        self.output_area.setFont(font)
        self.output_area.setObjectName('output_keyboard')
        self.output_area.setFixedHeight(120)
        self.output_layout.addWidget(self.output_area)

        # self.candidate = QWidget(objectName='candidate')
        # self.candidate.setFixedSize(1200, 40)
        # self.candidate_layout = QHBoxLayout(self.candidate)
        # self.candidate_layout.setContentsMargins(1, 0, 1, 0)
        # self.candidate_layout.setSpacing(20)
        # self.output_layout.addWidget(self.candidate)
        #
        # self.candidate_lst = []
        # for i in range(8):
        #     rect = StiRect('', self, config.candidateStiList[i], fontSize=30)
        #     rect.setFixedWidth(100)
        #     self.candidate_layout.addWidget(rect)
        #     self.candidate_lst.append(rect)
        #
        # self.candidate_layout.addSpacerItem(QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Maximum))
        # leftSti = StiRect('←', self, config.candidateStiList[8], fontSize=30)
        # leftSti.setFixedWidth(80)
        # self.candidate_layout.addWidget(leftSti)
        # self.candidate_lst.append(leftSti)
        #
        # rightSti = StiRect('→', self, config.candidateStiList[9], fontSize=30)
        # rightSti.setFixedWidth(80)
        # self.candidate_layout.addWidget(rightSti)
        # self.candidate_lst.append(rightSti)
        #
        # for rect in self.candidate_lst:
        #     rect.hide()

        rect = StiRect('Del', self, 15.6)
        rect.setGeometry(1355, 21, 200, 100)
        self.sti_rects[sti_idx] = rect
        sti_idx += 1
        h1.addSpacing(20)
        h1.addWidget(self.output_widget)
        h1.addSpacing(330)

        self.layout.addSpacing(5)

        self.progress = ProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.layout.addWidget(self.progress)
        self.layout.addSpacing(40)

        h2 = QHBoxLayout()
        h2.setContentsMargins(0, 0, 0, 0)
        h2.setSpacing(0)
        self.layout.addLayout(h2)
        self.stimulate_area = QWidget(objectName='stimulate_area')
        self.stimulate_area.setMouseTracking(True)
        h2.addSpacing(20)
        h2.addWidget(self.stimulate_area)

        font = QFont()
        font.setPixelSize(120)
        width, height = 110, 110
        interval = 50
        dev = 0

        for row in range(len(self.char_list)):
            row_chars = self.char_list[row].copy()
            if row_chars[0] == 'back':
                row_chars.pop(0)
            elif row_chars[0] == 'A':
                dev = width / 2
            elif row_chars[0] == 'Z':
                dev = width / 2 + width + interval
            elif row_chars[0] == 'Space':
                dev = width * 3
                width *= 8

            for col in range(len(row_chars)):
                char = row_chars[col]
                rect = StiRect(char, self.stimulate_area, self.sti_lst[sti_idx - 1])

                rect.setGeometry(int(col * (width + interval) + dev), int(row * (height + interval)), int(width), int(height))
                self.sti_rects[sti_idx] = rect
                sti_idx += 1

        # self.setContextMenuPolicy(Qt.CustomContextMenu)
        # self.customContextMenuRequested.connect(self.rightMenuShow)
        self.menu = RoundMenu()

        self.startAction = QAction('开始', triggered=self.start_sti_event)
        # self.dataSelectAction = QAction('开始数据采集', triggered=self.userInputEvent)

        self.menu.addAction(self.startAction)
        # self.menu.addAction(self.dataSelectAction)

        # self.userCreater = UserCreater()
        # self.userCreater.stimEnableSignal.connect(self.startDataCollectEvent)
        # self.userCreater.loadUserSignal.connect(self.start_sti_event)


    def flickSingleRect(self):
        fps = 60
        ifi = 1 / fps
        frameNum = 0
        phase = 0
        start_time = time.time()

        while self.collect_flag and self.currentItem is not None:
            self.currentItem.changeColor(self.currentItem.sti, frameNum, phase * np.pi, ifi)

            end_time = time.time()
            frameNum = (end_time - start_time) * fps
            time.sleep(0.000001)

    def rightMenuShow(self, a0):
        self.menu.exec(QCursor.pos())

    def set_result(self, idx):
        char = self.char_list_all[idx]
        if char == 'back':
            # if len(self.candidate_chars) > 0:
            #     self.candidate_chars.pop(-1)
            if len(self.show_chars) > 0:
                self.show_chars.pop(-1)
            # pyttsx3.speak("delete")
        elif char == 'Space':
            self.show_chars.append(' ')
            # pyttsx3.speak("space")
        else:
            # try:
            #     char = int(char)
            #     self.candidate_chars = []
            #     for rect in self.candidate_lst:
            #         rect.changeText('')
            #         rect.hide()
            # except:
            #     self.candidate_chars.append(char)
            #     pinyin_list = ''.join(c for c in self.candidate_chars)
            #     chineseResult = dag(self.dagParams, [pinyin_list], path_num=100, log=False)
            #     idx = 0
            #     while len(chineseResult) != 0:
            #         for rect in self.candidate_lst:
            #             rect.show()
            #         s = chineseResult.pop(0)
            #         self.candidate_lst[idx].changeText(s)
            #         idx += 1

            # pyttsx3.speak(str(char))
            self.show_chars.append(str(char))
        result = ''.join(c for c in self.show_chars)
        self.output_area.setText(result)

        if len(self.show_chars) > 30:
            self.show_chars = []


    # def setRecongnizeResult(self, result):
    #     for idx, value in enumerate(self.target_template_dict.keys()):
    #         if (idx + 1) == result:
    #             result = value
    #             break
    #
    #     self.result_times = 0
    #     index = []
    #     self.hide_index = []
    #     self.set_result(result)

    def getCurrentStimulate(self):
        if len(self.readty_sti_list) != 0:
            return self.readty_sti_list[-1]

    def setDefaultColor(self):
        for rect in self.sti_rects.values():
            rect.setDefaultColor()
        # QApplication.processEvents()

    # def startByUserEvent(self):
    #     if config.currentUser is None and not self.start_flick:
    #         self.userCreater.show()
    #     else:
    #         config.currentUser = None
    #         config.save()
    #         self.start_sti_event()

    def start_sti_event(self):
        if not self.start_flick:
            self.start_flick = True
            self.finish = False
            self.start_cache = False
            self.cache_data = np.array([])
            self.show_chars = []
            self.output_area.setText('')
            self.startAction.setText('结束')
            th = threading.Thread(target=self.flick)
            th.start()

        else:
            self.start_flick = False
            self.finish = True
            self.start_cache = False
            self.setDefaultColor()
            self.startAction.setText('开始')

    # def start_sti_event(self):
    #     if not self.start_flick:
    #         self.start_flick = True
    #         self.finish = False
    #         self.cache_data = np.array([])
    #         self.show_chars = []
    #         self.output_area.setText('')
    #         th = threading.Thread(target=self.flick)
    #         th.start()
    #         self.startAction.setText('结束')
    #     else:
    #         self.start_flick = False
    #         self.finish = True
    #         self.setDefaultColor()
    #         self.startAction.setText('开始')


    def flick(self):
        fps = 240
        ifi = 1 / fps
        frameNum = 0
        flag = True

        while not self.finish:
            start_time = time.time()
            now_time = 0
            while now_time < 3:
                now_time = time.time() - start_time
                self.progress.setVal(((3 - now_time) / 3) * 100)
                time.sleep(0.00000001)

            start_time = time.time()
            self.cache_data = np.array([])
            self.start_cache = True
            while self.cache_data.shape[-1] < self.times * 250 and not self.finish:
                end_time = time.time()
                for sti, rect in self.sti_rects.items():
                    rect.changeColor(self.sti_lst[sti - 1], end_time - start_time)

                self.progress.setVal((self.cache_data.shape[-1] / (self.times * 250)) * 100)
                time.sleep(0.00001)

            self.progress.setVal(100)
            self.setDefaultColor()
            self.start_cache = False
            if not self.finish and len(self.cache_data) != 0:
                used_data = self.cache_data[:, -1000:]
                self.cache_data = np.array([])
                result = self.fbcca.classify(used_data)
                result = int(result)
                self.set_result(result)

            # frameNum = (end_time - start_time) * fps
            # QApplication.processEvents()

    # def userInputEvent(self):
    #     if config.currentUser is None:
    #         self.userCreater.show()
    #     else:
    #         config.currentUser = None
    #         config.save()
    #         self.startDataCollectEvent()

    # def startDataCollectEvent(self):
    #     if self.collect_flag:
    #         self.collect_flag = False
    #         self.currentItem = None
    #         self.finish = True
    #         self.currentCollectSti = None
    #         self.startCollect = False
    #         self.setDefaultColor()
    #
    #         self.dataSelectAction.setText('开始数据采集')
    #     else:
    #         self.collect_flag = True
    #         self.finish = False
    #         self.dataSelectAction.setText('结束数据采集')
    #         self.currentCollectSti = 1
    #         th = threading.Thread(target=self.collectDataEvent)
    #         th.start()


    # def timeCutdown(self):
    #     rangeTimes = 3 * 100
    #     rect = self.sti_rects[self.currentCollectSti]
    #     start_time = time.time()
    #
    #     for i in range(1, 101):
    #         if not self.collect_flag:
    #             return
    #         end_time = time.time()
    #
    #         rect.changeColor(1, end_time - start_time)
    #
    #         self.progress.setVal(100 - i)
    #         time.sleep(0.03)

    # def collectDataEvent(self):
    #     while self.collect_flag:
    #         self.cache_data = np.array([])
    #         self.setDefaultColor()
    #         self.timeCutdown()
    #
    #         fps = 240
    #         ifi = 1 / fps
    #         frameNum = 0
    #         phase = 0
    #         start_time = time.time()
    #
    #         self.startCollect = True
    #         while self.cache_data.shape[-1] < (250 * 6) and self.collect_flag:
    #             end_time = time.time()
    #             for sti, rect in self.sti_rects.items():
    #                 # if sti >= 1 and sti <= 11:
    #                 #     phase = 0
    #                 # elif sti >= 12 and sti <= 21:
    #                 #     phase = 0.5
    #                 # elif sti >= 22 and sti <= 30:
    #                 #     phase = 1
    #                 # elif sti >= 31 and sti <= 37:
    #                 #     phase = 1.5
    #                 # elif sti >= 38:
    #                 #     phase = 0
    #                 # rect.changeColor(rect.sti, frameNum, phase * np.pi, ifi)
    #                 rect.changeColor(self.sti_lst[sti - 1], end_time - start_time)
    #
    #
    #             # end_time = time.time()
    #             # frameNum = (end_time - start_time) * fps
    #             time.sleep(0.000001)
    #
    #         self.startCollect = False
    #
    #         if config.currentUser is not None:
    #             savePath = os.path.join(config.userInfoPath, config.currentUser, 'data')
    #             print(savePath, self.char_list_all[self.currentCollectSti - 1], self.cache_data.shape)
    #             saveStiPath = os.path.join(savePath, str(self.sti_lst[self.currentCollectSti - 1]))
    #
    #             if not os.path.exists(saveStiPath):
    #                 os.makedirs(saveStiPath)
    #
    #             existedFiles = glob(os.path.join(saveStiPath, "*.mat"))
    #             savemat(os.path.join(saveStiPath, rf'{len(existedFiles)}.mat'), {'data': self.cache_data,
    #                                                                              'sti': self.sti_lst[self.currentCollectSti - 1]})
    #
    #         if self.currentCollectSti is not None:
    #             self.currentCollectSti += 1
    #             if self.currentCollectSti > len(self.char_list_all):
    #                 self.startDataCollectEvent()
    #                 recognition.train(config.currentUser)

    def setTrainingProcessEvent(self, val):
        self.progress.setVal(val)

    def getData(self, data):
        if self.finish:
            return

        if self.startCollect:
            if len(self.cache_data) == 0:
                self.cache_data = data
            else:
                self.cache_data = np.concatenate([self.cache_data, data], axis=-1)
            self.progress.setVal((100 / 6) * (self.cache_data.shape[-1] / 250))

        if self.start_cache:
            if len(self.cache_data) == 0:
                self.cache_data = data
            else:
                self.cache_data = np.concatenate([self.cache_data, data], axis=-1)
            # self.progress.setVal((self.cache_data.shape[-1] / 1000) * 100)

            # if self.cache_data.shape[-1] >= 1000:
            #     used_data = self.cache_data[:, -1000:]
            #     self.cache_data = np.array([])
            #     # th = threading.Thread(target=self.fbcca.classify, args=([used_data]))
            #     # th.start()
            #     result = self.fbcca.classify(used_data)
            #     # result, prob = recognition.predict(used_data)
            #     result = int(result)
            #     self.set_result(result)

        # if len(self.cache_data) >= 6:
        #     th = threading.Thread(target=self.recongnizeSSVEP, args=([self.cache_data]))
        #     th.start()
        #     # result = self.recongnizeSSVEP(self.cache_data)
        #     self.cache_data = []
        #     # print(result)

