from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from qfluentwidgets import *

class DeviceStateArea(QWidget):
    def __init__(self):
        super().__init__()
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)
        self.setObjectName('deviceStateArea')
        self.setQss()


        self.pointSize = 40
        self.channel_state = dict(
            channel_8   =  dict(impedance=999, fallFlag=-1),
            channel_2   =  dict(impedance=999, fallFlag=-1),
            channel_gnd =  dict(impedance=999, fallFlag=-1),
            channel_1   =  dict(impedance=999, fallFlag=-1),
            channel_3   =  dict(impedance=999, fallFlag=-1),
            channel_5   =  dict(impedance=999, fallFlag=-1),
            channel_ref =  dict(impedance=999, fallFlag=-1),
            channel_6   =  dict(impedance=999, fallFlag=-1),
            channel_4   =  dict(impedance=999, fallFlag=-1),
            channel_7   =  dict(impedance=999, fallFlag=-1),
        )
        self.device_elec = 0

    def setQss(self):
        with open(f'source/qss/mainWindow.qss', encoding='utf-8') as f:
            self.setStyleSheet(f.read())

    def change_device_state(self, elecValue, impedance, fallFlag):
        self.device_elec = elecValue
        channel_names = [8, 7, 6, 5, 4, 3, 2, 1]
        flag = False
        for idx, channel_name in enumerate(channel_names):
            self.channel_state[f'channel_{channel_name}']['fallFlag'] = fallFlag[idx]
            self.channel_state[f'channel_{channel_name}']['impedance'] = impedance[idx]

            if fallFlag[idx] == 0:
                flag = True

        if not flag:
            self.channel_state['channel_gnd']['fallFlag'] = -1
            self.channel_state['channel_ref']['fallFlag'] = -1
            self.channel_state['channel_gnd']['impedance'] = 999
            self.channel_state['channel_ref']['impedance'] = 999
        else:
            self.channel_state['channel_gnd']['fallFlag'] = 0
            self.channel_state['channel_ref']['fallFlag'] = 0
            self.channel_state['channel_gnd']['impedance'] = 0
            self.channel_state['channel_ref']['impedance'] = 0

        self.update()


    def paintEvent(self, event):
        super().paintEvent(event)

        width = self.width()
        height = self.height()

        start_x = 10
        start_y = 10
        self.channel_pos_list = dict(
            channel_8   = QPointF(width * 0.05 - self.pointSize / 2, height * 0.15 - self.pointSize / 2),
            channel_2   = QPointF(width * 0.05 - self.pointSize / 2, height * 0.85 - self.pointSize / 2),
            channel_gnd = QPointF(width * 0.15 - self.pointSize / 2, height * 0.25 - self.pointSize / 2),
            channel_1   = QPointF(width * 0.17 - self.pointSize / 2, height * 0.75 - self.pointSize / 2),
            channel_3   = QPointF(width * 0.22 - self.pointSize / 2, height * 0.2 - self.pointSize / 2),
            channel_5   = QPointF(width * 0.22 - self.pointSize / 2, height * 0.8 - self.pointSize / 2),
            channel_ref = QPointF(width * 0.29 - self.pointSize / 2, height * 0.25 - self.pointSize / 2),
            channel_6   = QPointF(width * 0.27 - self.pointSize / 2, height * 0.75 - self.pointSize / 2),
            channel_4   = QPointF(width * 0.39 - self.pointSize / 2, height * 0.15 - self.pointSize / 2),
            channel_7   = QPointF(width * 0.39 - self.pointSize / 2, height * 0.85 - self.pointSize / 2),
        )

        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)

        for channel_name, channel_pos in self.channel_pos_list.items():
            if self.channel_state[channel_name]['fallFlag'] == 0:
                painter.setPen(QPen(Qt.green, 2, Qt.SolidLine))
                flag = True
            else:
                painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
                flag = False

            if flag:
                impedance = self.channel_state[channel_name]['impedance']
                color = int((impedance / 100) * 255)
                painter.setBrush(QColor(0, 255 - color, 0))
            else:
                painter.setBrush(Qt.red)

            rect = QRectF(channel_pos.x(), channel_pos.y(), self.pointSize, self.pointSize)
            painter.drawEllipse(rect)

            ch_name = channel_name.split('_')[-1]

            if flag:
                painter.setPen(QPen(Qt.black, 1, Qt.SolidLine))
            else:
                painter.setPen(QPen(Qt.white, 1, Qt.SolidLine))
            painter.setFont(QFont("simsun", 12))
            painter.drawText(rect, Qt.AlignCenter, ch_name)

        rect = QRectF(width - 220, 20, 200, 20)
        painter.setPen(QPen(Qt.white, 2, Qt.SolidLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)

        current_elec = QRectF(width - 220, 20, self.device_elec * 2, 20)
        painter.setPen(Qt.NoPen)
        painter.setBrush(Qt.green)
        painter.drawRect(current_elec)

