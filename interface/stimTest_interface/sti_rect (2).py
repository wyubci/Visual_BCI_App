from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import numpy as np

class StiRect(QLabel):
    def __init__(self, text, parent, color=255):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)
        # self.setObjectName('fangxiang')

        # font = QFont()
        # font.setPixelSize(120)
        # self.setFont(font)
        # self.setText(text)

        self.text = text
        self.color_value = color
        self.current_color = QColor(color, color, color)
        self.default_color = QColor(color, color, color)

    def flicker(self, f, i, phi, ifi):
        freq = f
        angfreq = 2 * np.pi * freq
        light = 255 * ((1 + np.sin(angfreq * (i * ifi) + phi)) / 2)
        return int(light)

    def changeColor(self, f, i, phi, ifi):
        color = self.flicker(f, i, phi, ifi)
        # qss = '#fangxiang{background-color: ' + f'rgb({color}, {color}, {color})' + '}'
        # self.setStyleSheet(qss)
        self.color_value = color
        self.current_color = QColor(color, color, color)
        self.update()


    def selectState(self):
        if self.current_color == self.default_color:
            self.current_color = QColor(255, 0, 0)
        else:
            self.current_color = self.default_color
        self.color_value = 255
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
        painter.drawEllipse(self.rect().adjusted(1, 1, -1, -1))

        painter.setBrush(QBrush())
        pen = QPen()
        pen.setWidth(2)
        pen.setColor(QColor(255, 0, 0, self.color_value))
        painter.setPen(pen)
        painter.drawEllipse(self.rect().adjusted(1, 1, -1, -1))

        text_color = QColor(0, 0, 0)
        font = QFont()
        font.setPixelSize(60)

        pen = QPen()
        pen.setColor(text_color)
        pen.setBrush(text_color)

        painter.setPen(pen)
        painter.setFont(font)

        painter.drawText(self.rect().adjusted(1, 1, -1, -1), Qt.AlignCenter, self.text)

        painter.end()
