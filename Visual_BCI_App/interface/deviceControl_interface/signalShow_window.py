import numpy as np
from PyQt5.QtChart import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import os
import threading

class EEGSignalShowModule(QChartView):
    def __init__(self, ch_name):
        super(EEGSignalShowModule, self).__init__()
        self.setObjectName('eegSignalShowModule')
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)
        self.ch_name = ch_name
        self.flag = True
        self.max_time = 125
        self.y_value = 1e7
        self.setQss()

        self.points = []
        self.count = 0
        self.flag = False

        self.__initChart__()

    def setQss(self):
        with open(f'source/qss/mainWindow.qss', encoding='utf-8') as f:
            self.setStyleSheet(f.read())

    def __initChart__(self):
        self.chart = QChart()
        self.chart.setTitle(self.ch_name)
        self.chart.legend().hide()
        self.chart.setAnimationOptions(QChart.SeriesAnimations)
        self.setRenderHint(QPainter.Antialiasing)
        self.setChart(self.chart)
        self.chart.setMargins(QMargins(0, 0, 0, 0))
        self.setContentsMargins(0, 0, 0, 0)
        self.setViewportMargins(0, 0, 0, 0)

        self.axisX = QValueAxis()
        self.axisY = QValueAxis()
        self.axisX.setGridLineVisible(False)
        self.axisY.setGridLineVisible(False)
        self.axisX.setRange(0, self.max_time)
        self.axisY.setRange(-self.y_value, self.y_value)
        self.axisX.setTickCount(1)
        # self.axisY.setTickCount(0.1)

        # self.chart.setAxisX(self.axisX)
        # self.chart.setAxisY(self.axisY)

        self.lineSeries = QSplineSeries()
        self.lineSeries.setUseOpenGL(True)
        self.chart.addSeries(self.lineSeries)

        self.chart.addAxis(self.axisX, Qt.AlignBottom)
        self.chart.addAxis(self.axisY, Qt.AlignLeft)
        self.lineSeries.attachAxis(self.axisX)
        self.lineSeries.attachAxis(self.axisY)

        self.chart.setBackgroundBrush(QBrush(QColor(34, 36, 42)))
        self.chart.setTitleBrush(QBrush(Qt.white))
        self.chart.axisX().setLabelsColor(Qt.white)
        self.chart.axisY().setLabelsColor(Qt.white)

    def set_data(self, data_all):
        min_value = np.min(data_all, axis=-1)
        max_value = np.max(data_all, axis=-1)

        self.axisY.setMin(min_value)
        self.axisY.setMax(max_value)
        # self.lineSeries.replace(data_all)
        # points = []
        # count = self.lineSeries.count()
        # if self.count >= 250:
        #     self.count = 0
        #     self.flag = True
        self.points = []
        for i in range(data_all.shape[0]):
            self.points.append(QPointF(i, data_all[i]))

        # print(data_all.shape)
        # if len(self.points) < 250:
        #     for i in range(data_all.shape[0]):
        #         self.points.append(QPointF(self.count, data_all[i]))
        #         self.count += 1
        # else:
        #     if self.count >= 250:
        #         self.count = 0
        #     for i in range(data_all.shape[0]):
        #         self.points[self.count] = QPointF(self.count, data_all[i])
        #         self.count += 1

        # for i in range(data_all.shape[0]):
        #     if len(self.points) == 250:
        #         self.points.pop(0)
        #
        #     self.points.append(QPointF(len(self.points), data_all[i]))
            # self.count += 1

        # if count < 250:
        #     self.lineSeries.append(self.points)
        #     self.clearData()
        #     count = self.lineSeries.count()
        # print(len(self.points))
        self.lineSeries.replace(self.points)
        self.update()

    def clearData(self):
        self.lineSeries.clear()