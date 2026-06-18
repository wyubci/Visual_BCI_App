from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import numpy as np

class StiRect(QLabel):
    def __init__(self, text, parent, sti, fontSize=60, color=255):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)
        # self.setObjectName('fangxiang')

        # font = QFont()
        # font.setPixelSize(120)
        # self.setFont(font)
        # self.setText(text)
        self.fontSize = fontSize

        self.sti = sti

        self.text = text
        self.color_value = color
        self.current_color = QColor(color, color, color, 255)
        self.default_color = QColor(color, color, color, 255)

    # def flicker(self, f, i, phi, ifi):
    #     freq = f
    #     angfreq = 2 * np.pi * freq
    #     light = 255 * ((1 + np.sin(angfreq * (i * ifi) + phi)) / 2)
    #     return int(light)

    def changeText(self, text):
        self.text = text
        self.update()

    def flicker(self, freq, now_time):
        light = 255 * np.sin(2 * np.pi * now_time * freq)
        return int(light)

    def changeColor(self, f, now_time):
        # color = self.flicker(f, i, phi, ifi)
        color = self.flicker(f, now_time)

        self.color_value = color
        self.current_color = QColor(255, 255, 255, color)
        self.update()

    def setDefaultColor(self):
        # qss = '#fangxiang{background-color: ' + f'rgb({0}, {0}, {0})' + '}'
        # self.setStyleSheet(qss)
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

        painter.setBrush(QBrush())
        pen = QPen()
        pen.setWidth(2)
        pen.setColor(QColor(255, 0, 0))
        painter.setPen(pen)
        # painter.drawEllipse(self.rect().adjusted(1, 1, -1, -1))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 5, 5)

        text_color = QColor(0, 0, 0)
        font = QFont()
        font.setPixelSize(self.fontSize)

        pen = QPen()
        pen.setColor(text_color)
        pen.setBrush(text_color)

        painter.setPen(pen)
        painter.setFont(font)

        painter.drawText(self.rect().adjusted(1, 1, -1, -1), Qt.AlignCenter, self.text)

        painter.end()
