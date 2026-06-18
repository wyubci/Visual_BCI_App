from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import numpy as np
from qfluentwidgets import *

class DoubleStiRect(QWidget):
    def __init__(self, text, left_sti, right_sti, fontSize=60, color=255):
        super().__init__()
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)

        self.text = text
        self.fontSize = fontSize

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(1)
        self.setLayout(self.layout)


        self.hlayout = QHBoxLayout()
        self.hlayout.setContentsMargins(2, 2, 2, 2)
        self.hlayout.setSpacing(0)
        self.layout.addLayout(self.hlayout)

        self.rect1 = StiRect(self, left_sti, color)
        self.rect2 = StiRect(self, right_sti, color)
        self.line = QLabel(objectName='line')
        self.hlayout.addWidget(self.rect1)
        self.hlayout.addWidget(self.line)
        self.hlayout.addWidget(self.rect2)

        self.label = BodyLabel()
        self.label.setText(self.text)
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label)

        self.centerV = QLabel(objectName='centerV', parent=self)
        self.centerH = QLabel(objectName='centerH', parent=self)

    def setQss(self):
        with open(f'source/qss/mainWindow.qss', encoding='utf-8') as f:
            self.setStyleSheet(f.read())

    def resizeEvent(self, event):
        super().resizeEvent(event)

        self.label.setFixedHeight(20)

        w = self.width() - 5
        h = self.height() - 24
        self.rect1.setFixedSize(w // 2, h)
        self.line.setFixedSize(1, h - 10)
        self.rect2.setFixedSize(w // 2, h)

        self.centerV.setGeometry(self.width() // 2 - 1, h // 2 - 15, 2, 30)
        self.centerH.setGeometry(self.width() // 2 - 15, h // 2 - 1, 30, 2)

    def changeColor(self, now_time):
        self.rect1.changeColor(self.rect1.sti, now_time)
        self.rect2.changeColor(self.rect2.sti, now_time)

        self.update()

    def setDefaultColor(self):
        self.rect1.setDefaultColor()
        self.rect2.setDefaultColor()
        self.update()

    def paintEvent(self, a0):
        super().paintEvent(a0)

        painter = QPainter()
        painter.begin(self)

        painter.setBrush(QBrush())
        pen = QPen()
        pen.setWidth(2)
        pen.setColor(QColor(255, 0, 0))
        painter.setPen(pen)
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -20), 5, 5)

        painter.end()


class StiRect(QLabel):
    def __init__(self, parent, sti, color=255):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)
        self.sti = sti

        self.color_value = color
        self.current_color = QColor(color, color, color, 255)
        self.default_color = QColor(color, color, color, 255)

    def flicker(self, freq, now_time):
        light = 255 * np.sin(2 * np.pi * now_time * freq)
        return int(light)

    def changeColor(self, f, now_time):
        color = self.flicker(f, now_time)

        self.color_value = color
        self.current_color = QColor(255, 255, 255, color)
        self.update()

    def setDefaultColor(self):
        self.current_color = self.default_color
        self.color_value = 255
        self.update()


    def paintEvent(self, a0):
        super().paintEvent(a0)

        painter = QPainter()
        painter.begin(self)
        painter.setRenderHints(QPainter.Antialiasing)
        painter.setPen(self.current_color)
        painter.setBrush(self.current_color)
        # painter.drawEllipse(self.rect().adjusted(1, 1, -1, -1))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 5, 5)

        # painter.setBrush(QBrush())
        # pen = QPen()
        # pen.setWidth(2)
        # pen.setColor(QColor(255, 0, 0))
        # painter.setPen(pen)
        # # painter.drawEllipse(self.rect().adjusted(1, 1, -1, -1))
        # painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 5, 5)

        # text_color = QColor(0, 0, 0)
        # font = QFont()
        # font.setPixelSize(self.fontSize)
        #
        # pen = QPen()
        # pen.setColor(text_color)
        # pen.setBrush(text_color)
        #
        # painter.setPen(pen)
        # painter.setFont(font)
        #
        # painter.drawText(self.rect().adjusted(1, 1, -1, -1), Qt.AlignCenter, self.text)

        painter.end()
