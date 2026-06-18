from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import numpy as np

class StiRect(QLabel):
    def __init__(self, fileName, parent, sti, text='', color=255):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)
        self.sti = sti
        self.text = text
        self.pix = QPixmap(fileName)
        self.color_value = color
        self.current_color = QColor(color, color, color, 255)
        self.default_color = QColor(color, color, color, 255)


    def changeText(self, text):
        self.text = text
        self.update()

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

        painter.setBrush(QBrush())
        pen = QPen()
        pen.setWidth(2)
        pen.setColor(QColor(255, 0, 0))
        painter.setPen(pen)
        # painter.drawEllipse(self.rect().adjusted(1, 1, -1, -1))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 5, 5)

        width = self.rect().width() / 4
        painter.drawPixmap(self.rect().adjusted(width, width, -width, -width), self.pix)

        painter.end()
