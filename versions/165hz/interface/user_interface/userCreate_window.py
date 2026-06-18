import numpy as np
from PyQt5.QtChart import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from qfluentwidgets import *
from qframelesswindow import *

import os

from config import config
from controller.recognition_controller import recognition

class UserCreater(FramelessWindow):
    stimEnableSignal = pyqtSignal()
    loadUserSignal = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)

        self.setTitleBar(StandardTitleBar(self))
        self.setWindowTitle('用户')
        self.titleBar.iconLabel.hide()

        self.setQss()

        self.mode = 'new'

        self.__initLayout__()
        self.__initItems__()

        self.setFixedSize(300, 150)

    def setQss(self):
        with open(f'source/qss/mainWindow.qss', encoding='utf-8') as f:
            self.setStyleSheet(f.read())


    def __initLayout__(self):
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(5, self.titleBar.height(), 5, 5)
        self.layout.setSpacing(0)
        self.setLayout(self.layout)

    def __initItems__(self):
        self.userNameEdit = LineEdit()
        self.userNameEdit.setPlaceholderText('请输入用户名')
        self.layout.addWidget(self.userNameEdit)

        self.layout1 = QHBoxLayout()
        self.layout1.setContentsMargins(0, 0, 0, 0)
        self.layout1.setSpacing(0)
        self.layout.addLayout(self.layout1)
        self.confirm_button = PushButton()
        self.confirm_button.setText('确认')
        self.confirm_button.clicked.connect(self.confirmEvent)
        self.layout1.addWidget(self.confirm_button, alignment=Qt.AlignRight)

    def confirmEvent(self):
        userName = self.userNameEdit.text()

        if userName == '':
            self.loadUserSignal.emit()
            self.close()
            return

        userPackage = os.path.join(config.userInfoPath, userName)
        userWeightsPackage = os.path.join(config.userInfoPath, userName, 'weight')
        userDataPackage = os.path.join(config.userInfoPath, userName, 'data')

        if not os.path.exists(userPackage):
            os.makedirs(userPackage)
            os.makedirs(userWeightsPackage)
            os.makedirs(userDataPackage)

        config.currentUser = userName
        config.save()
        self.close()

        if os.path.exists(os.path.join(userWeightsPackage, 'best.pth')):
            recognition.load_state_dict(userName)
            self.loadUserSignal.emit()
        else:
            self.stimEnableSignal.emit()
