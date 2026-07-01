from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from qfluentwidgets import *


class DeviceStateArea(QWidget):
    """设备状态显示区域。

    - LSL 模式：显示 32 导信号质量热力图（绿/黄/红）
    - NeuroDance 模式：显示 8 电极阻抗 + 脱落状态
    """

    def __init__(self):
        super().__init__()
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)
        self.setObjectName('deviceStateArea')
        self.setQss()

        self.pointSize = 20

        # LSL 模式
        self._lsl_mode = False
        self._quality_metrics = []       # list[dict] from _computeSignalQuality

        # NeuroDance 模式（保留向后兼容）
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
        with open('source/qss/mainWindow.qss', encoding='utf-8') as f:
            self.setStyleSheet(f.read())

    def change_device_state(self, elecValue_or_quality, impedance, fallFlag):
        """接收设备状态信号。

        LSL 模式：elecValue_or_quality 是 list[dict]（每通道信号质量）
        NeuroDance 模式：elecValue_or_quality 是 int（电量），
                        impedance 是 list[int]（阻抗值），
                        fallFlag 是 list[int]（脱落标志）
        """
        if isinstance(elecValue_or_quality, list) and len(elecValue_or_quality) > 0:
            # LSL 质量指标模式
            self._lsl_mode = True
            self._quality_metrics = elecValue_or_quality
        else:
            # NeuroDance 传统模式
            self._lsl_mode = False
            self.device_elec = elecValue_or_quality if isinstance(elecValue_or_quality, int) else 0

            if impedance is not None and fallFlag is not None:
                channel_names = [8, 7, 6, 5, 4, 3, 2, 1]
                flag_detected = False
                for idx, channel_name in enumerate(channel_names):
                    if idx < len(fallFlag):
                        self.channel_state[f'channel_{channel_name}']['fallFlag'] = fallFlag[idx]
                    if idx < len(impedance):
                        self.channel_state[f'channel_{channel_name}']['impedance'] = impedance[idx]
                    if idx < len(fallFlag) and fallFlag[idx] == 0:
                        flag_detected = True

                if not flag_detected:
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

        if self._lsl_mode:
            self._paintLslHeatmap(event)
        else:
            self._paintNeuroDanceHeadcap(event)

    # ------------------------------------------------------------------
    # LSL 32 导信号质量热力图
    # ------------------------------------------------------------------

    def _paintLslHeatmap(self, event):
        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        n_channels = len(self._quality_metrics)

        if n_channels == 0:
            painter.end()
            return

        cols = 8
        rows = max(1, (n_channels + cols - 1) // cols)
        cell_w = (width - 20) // cols
        cell_h = min((height - 40) // rows, cell_w)
        grid_w = cell_w * cols
        grid_h = cell_h * rows
        start_x = (width - grid_w) // 2
        start_y = max(10, (height - grid_h - 30) // 2)

        for i in range(n_channels):
            col = i % cols
            row = i // cols
            x = start_x + col * cell_w + 2
            y = start_y + row * cell_h + 2
            w = cell_w - 4
            h_rect = cell_h - 4

            m = self._quality_metrics[i]
            color = self._quality_color(m)

            painter.setBrush(color)
            painter.setPen(QPen(QColor(60, 60, 60), 1))
            rect = QRectF(x, y, w, h_rect)
            painter.drawRoundedRect(rect, 4, 4)

            # 通道编号
            painter.setPen(QPen(Qt.white, 1))
            font = QFont("Consolas", 8)
            painter.setFont(font)
            painter.drawText(rect, Qt.AlignCenter, str(i))

        # 图例
        legend_y = start_y + grid_h + 5
        painter.setFont(QFont("SimSun", 10))
        for idx, (label, clr) in enumerate([
            ('好', QColor(0, 180, 0)),
            ('警告', QColor(220, 180, 0)),
            ('差', QColor(200, 50, 50)),
            ('未选中', QColor(80, 80, 80)),
        ]):
            lx = start_x + idx * 100
            painter.setBrush(clr)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(QRectF(lx, legend_y, 14, 14), 3, 3)
            painter.setPen(QPen(Qt.white, 1))
            painter.drawText(QRectF(lx + 18, legend_y, 60, 14), Qt.AlignVCenter, label)

        painter.end()

    @staticmethod
    def _quality_color(metric):
        """根据信号质量返回颜色。"""
        var = metric.get('variance', 0)
        noise = metric.get('noise_50hz_ratio', 0)
        sat = metric.get('is_saturated', 0)

        # 饱和 → 红
        if sat > 0.5:
            return QColor(200, 50, 50)
        # 平线（脱落）→ 红
        if var < 1e-6:
            return QColor(200, 50, 50)
        # 高 50Hz 噪声 → 黄
        if noise > 0.3:
            return QColor(220, 180, 0)
        # 信号良好 → 绿
        return QColor(0, 180, 0)

    # ------------------------------------------------------------------
    # NeuroDance 8 电极头帽视图（保留兼容）
    # ------------------------------------------------------------------

    def _paintNeuroDanceHeadcap(self, event):
        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()

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

        for channel_name, channel_pos in self.channel_pos_list.items():
            if self.channel_state[channel_name]['fallFlag'] == 0:
                painter.setPen(QPen(Qt.green, 2, Qt.SolidLine))
                flag = True
            else:
                painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
                flag = False

            if flag:
                impedance = self.channel_state[channel_name]['impedance']
                color_val = int((impedance / 100) * 255)
                painter.setBrush(QColor(0, 255 - min(color_val, 255), 0))
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

        # 电量条
        rect = QRectF(width - 220, 20, 200, 20)
        painter.setPen(QPen(Qt.white, 2, Qt.SolidLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)

        current_elec = QRectF(width - 220, 20, self.device_elec * 2, 20)
        painter.setPen(Qt.NoPen)
        painter.setBrush(Qt.green)
        painter.drawRect(current_elec)

        painter.end()
