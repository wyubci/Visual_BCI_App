from PyQt5.QtCore import *
from PyQt5.QtGui import *
from qfluentwidgets import *
from qframelesswindow import *
from qfluentwidgets import FluentIcon as FIF
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.params_controller import paramsController
from interface.deviceControl_interface.deviceControl_window import DeviceControlWindow
from interface.drone_interface.drone_window import DroneControlWindow
from interface.car_interface.car_window import CarControlWindow


class HomeWindow(MSFluentWindow):
    def __init__(self):
        super().__init__()
        self.setObjectName('HomeWindow')
        self.setWindowTitle('视觉脑机接口控制系统')
        self.setWindowIcon(QIcon(str(PROJECT_ROOT / "source" / "icons" / "logo.png")))

        self.setQss()
        paramsController.current_window = 'home'
        self.__initItems__()
        self.stackedWidget.currentChanged.connect(self.interfaceChange)

    def interfaceChange(self, index):
        paramsController.current_window = 'home' if index == 0 else index

    def setQss(self):
        with open(PROJECT_ROOT / 'source' / 'qss' / 'mainWindow.qss', encoding='utf-8') as f:
            self.setStyleSheet(f.read())

    def __initItems__(self):
        self.deviceControlWindow = DeviceControlWindow(objectName='deviceControlWindow')
        self.droneControlWindow = DroneControlWindow(objectName='droneControlWindow')
        self.carControlWindow = CarControlWindow(objectName='carControlWindow')
        self.carTrainingWindow = self.carControlWindow.data_management_widget
        self.carTrainingWindow.setObjectName('carTrainingWindow')

        self.addSubInterface(self.deviceControlWindow, FIF.HOME, '设备控制')
        self.addSubInterface(self.droneControlWindow, FIF.ROBOT, '脑控 Drone')
        self.addSubInterface(self.carControlWindow, FIF.IOT, '脑控小车')
        self.addSubInterface(self.carTrainingWindow, FIF.MARKET, '小车训练/权重')

        self.deviceControlWindow.set_car_data_sink(self.carControlWindow.push_eeg_data_threadsafe)
        self.deviceControlWindow.signalSendSignal_drone.connect(self.droneControlWindow.getData)
        self.deviceControlWindow.signalSendSignal_car.connect(self.carControlWindow.getData)
