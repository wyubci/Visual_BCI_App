import threading

import numpy as np
from scipy.io import savemat
import math
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from qfluentwidgets import *
from qframelesswindow import *
from win32api import GetMonitorInfo, MonitorFromPoint
from qfluentwidgets import FluentIcon as FIF

from controller.params_controller import paramsController
# from controller.faceDetect_controller import FaceDetector

from interface.deviceControl_interface.deviceControl_window import DeviceControlWindow
from interface.keyboard_interface.keyboard_window import KeyboardControlWindow
from interface.stimTest_interface.stimTest_window import StimTestWindow
from interface.care_interface.care_window import CareControlWindow
from interface.dualFrequency_interface.dualFrequency_window import DualFrequencyWindow
from interface.drone_interface.drone_window import DroneControlWindow
from interface.car_interface.car_window import CarControlWindow

class HomeWindow(MSFluentWindow):
    def __init__(self):
        super().__init__()
        self.setObjectName('HomeWindow')
        self.setWindowTitle('视觉脑机接口控制系统')
        self.setWindowIcon(QIcon("source/icons/logo.png"))

        self.setQss()

        paramsController.current_window = 'home'

        self.__initItems__()

        self.stackedWidget.currentChanged.connect(self.interfaceChange)
        # self.faceThread = FaceDetector()
        # self.faceThread.start()


    def interfaceChange(self, a0):
        if a0 == 0:
            paramsController.current_window = 'home'
        else:
            paramsController.current_window = a0

    def setQss(self):
        with open(f'source/qss/mainWindow.qss', encoding='utf-8') as f:
            self.setStyleSheet(f.read())

    def __initItems__(self):
        self.deviceControlWindow = DeviceControlWindow(objectName='deviceControlWindow')
        self.keyboardWindow = KeyboardControlWindow(objectName='keyboardWindow')
        self.stimTestWindow = StimTestWindow(objectName='stimTestWindow')
        self.careControlWindow = CareControlWindow(objectName='careControlWindow')
        self.dualFrequencyWindow = DualFrequencyWindow(objectName='dualFrequencyWindow')
        self.droneControlWindow = DroneControlWindow(objectName='droneControlWindow')
        self.carControlWindow = CarControlWindow(objectName='carControlWindow')

        self.addSubInterface(self.deviceControlWindow, FIF.HOME, '设备控制')
        self.addSubInterface(self.stimTestWindow, FIF.MARKET, '数据分析')
        self.addSubInterface(self.keyboardWindow, FIF.CHAT, '脑控打字')
        self.addSubInterface(self.careControlWindow, FIF.FEEDBACK, '特护病床')
        self.addSubInterface(self.dualFrequencyWindow, FIF.FEEDBACK, '双频打字')
        self.addSubInterface(self.droneControlWindow, FIF.ROBOT, '脑控Drone')
        self.addSubInterface(self.carControlWindow, FIF.IOT, '脑控小车')

        self.deviceControlWindow.signalSendSignal_keyboard.connect(self.keyboardWindow.getData)
        self.deviceControlWindow.signalSendSignal_test.connect(self.stimTestWindow.getData)
        self.deviceControlWindow.signalSendSignal_care.connect(self.careControlWindow.getData)
        self.deviceControlWindow.signalSendSignal_dualFreq.connect(self.dualFrequencyWindow.getData)
        self.deviceControlWindow.signalSendSignal_drone.connect(self.droneControlWindow.getData)
        self.deviceControlWindow.signalSendSignal_car.connect(self.carControlWindow.getData)

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        if event.key() == Qt.Key_Return:
            if paramsController.current_window == 2:
                self.keyboardWindow.start_sti_event()
            elif paramsController.current_window == 3:
                self.careControlWindow.start_sti_event()

            elif paramsController.current_window == 4:
                self.dualFrequencyWindow.start_sti_event()