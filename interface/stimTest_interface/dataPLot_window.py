import numpy as np
from PyQt5.QtChart import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import os
import threading

class DataAnalysisPlotModule(QChartView):
    def __init__(self, ch_name):
        super(DataAnalysisPlotModule, self).__init__()
        self.setObjectName('eegSignalShowModule')
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)
        self.ch_name = ch_name
        self.flag = True
        self.max_time = 250
        # self.freq = np.linspace(0, 250, 300)

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

        self.points = []
        for i in range(data_all.shape[0]):
            self.points.append(QPointF(i, data_all[i]))

        self.lineSeries.replace(self.points)
        self.update()

    def clearData(self):
        self.lineSeries.clear()