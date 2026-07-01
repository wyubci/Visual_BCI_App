import time

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from qfluentwidgets import *
import serial
import serial.tools.list_ports
import threading
import ctypes
import numpy as np
import os
from scipy import signal

from .deviceState_area import DeviceStateArea
from .signalShow_window import EEGSignalShowModule
from controller.params_controller import paramsController
from config import config

# NeuroDance (保留向后兼容，安装失败时不阻塞)
try:
    from .nd_device_demo import NdDevice
except ImportError:
    NdDevice = None

# BrainVision RDA (首选) / LSL (备选)
from .rda_receiver import RdaReceiver
try:
    from .lsl_receiver import LslReceiver
except ImportError:
    LslReceiver = None

_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEVICE_DLL = os.path.join(_base_dir, "source", "device", "LinkMe.dll")
if os.path.isfile(_DEVICE_DLL):
    my_dll = ctypes.CDLL(_DEVICE_DLL)
else:
    my_dll = None

# init functions of dll (仅在 DLL 可用时注册)
if my_dll is not None:
    my_dll.dataProtocol.argtypes = (ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int)
    my_dll.dataProtocol.restype = ctypes.c_int
    my_dll.getElectricityValue.restype = ctypes.c_int
    my_dll.getFallFlag.argtypes = (ctypes.POINTER(ctypes.c_int), )
    my_dll.getData.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_double))
    my_dll.getDataCurrIndex.argtypes = (ctypes.POINTER(ctypes.c_long), )
    my_dll.getImpedance.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double) )

class DeviceControlWindow(QWidget):
    deviceStateSignal = pyqtSignal(object, object, object)
    signalSendSignal = pyqtSignal(object)
    signalSendSignal_drone = pyqtSignal(object)
    signalSendSignal_car = pyqtSignal(object)

    def __init__(self, objectName):
        super().__init__()
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)
        self.setObjectName(objectName)
        self.setQss()

        self.ser = None
        self.is_received_data = False
        self.lock = threading.Lock()
        self.nd_device = None
        self.rda_receiver = None
        self.car_data_sink = None

        self.data_buffer = []

        fs = 250
        f0 = 50
        q = 35
        self.bpB, self.bpA = signal.iircomb(f0, q, ftype='notch', fs=fs)
        self.bpB2, self.bpA2 = signal.butter(5, [4, 90], 'bandpass', fs=fs)
        self.__layout__()
        self.__items__()

    def setQss(self):
        with open(f'source/qss/mainWindow.qss', encoding='utf-8') as f:
            self.setStyleSheet(f.read())

    def __layout__(self):
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.setLayout(self.layout)

    def __items__(self):
        self.tools_area = QWidget()
        self.layout.addWidget(self.tools_area)
        self.tools_layout = QHBoxLayout(self.tools_area)
        self.tools_layout.setContentsMargins(10, 10, 10, 5)
        self.tools_layout.setSpacing(5)

        # 添加设备类型选择
        self.device_type_label = BodyLabel()
        self.device_type_label.setText('设备类型')
        self.device_type_combo = ComboBox()
        self.device_type_combo.addItems(['BrainVision RDA', 'NeuroDance串口', 'NeuroDance TCP'])
        # 根据 config 设置默认设备类型
        dt = getattr(config, 'device_type', 'lsl')
        if dt == 'lsl':
            self.device_type_combo.setCurrentIndex(0)
        elif dt in ('neuro_dance_serial', 'serial'):
            self.device_type_combo.setCurrentIndex(1)
        else:
            self.device_type_combo.setCurrentIndex(2)
        self.device_type_combo.currentIndexChanged.connect(self.onDeviceTypeChanged)
        self.tools_layout.addWidget(self.device_type_label)
        self.tools_layout.addWidget(self.device_type_combo)

        self.tools_layout.addSpacing(10)

        # ---- BrainVision RDA 设置 ----
        self.lsl_widget = QWidget()
        self.lsl_layout = QHBoxLayout(self.lsl_widget)
        self.lsl_layout.setContentsMargins(0, 0, 0, 0)
        self.lsl_layout.setSpacing(5)

        self.rda_host_label = BodyLabel()
        self.rda_host_label.setText('RDA Host')
        self.rda_host_edit = LineEdit()
        self.rda_host_edit.setText('127.0.0.1')
        self.rda_host_edit.setFixedWidth(100)
        self.lsl_layout.addWidget(self.rda_host_label)
        self.lsl_layout.addWidget(self.rda_host_edit)

        self.rda_port_label = BodyLabel()
        self.rda_port_label.setText('RDA Port')
        self.rda_port_edit = LineEdit()
        self.rda_port_edit.setText('51234')
        self.rda_port_edit.setFixedWidth(60)
        self.lsl_layout.addWidget(self.rda_port_label)
        self.lsl_layout.addWidget(self.rda_port_edit)

        self.lsl_channel_btn = PushButton()
        self.lsl_channel_btn.setText('选择通道')
        self.lsl_channel_btn.clicked.connect(self._showChannelSelectionDialog)
        self.lsl_layout.addWidget(self.lsl_channel_btn)

        self.lsl_channel_count_label = BodyLabel()
        sel = getattr(config, 'lsl_selected_channels', list(range(32)))
        self.lsl_channel_count_label.setText(f'已选 {len(sel)}/32 导')
        self.lsl_layout.addWidget(self.lsl_channel_count_label)

        self.tools_layout.addWidget(self.lsl_widget)
        self.lsl_widget.hide()  # 默认 NeuroDance TCP，隐藏 RDA 设置

        # ---- NeuroDance 串口设置 ----
        self.serial_widget = QWidget()
        self.serial_layout = QHBoxLayout(self.serial_widget)
        self.serial_layout.setContentsMargins(0, 0, 0, 0)
        self.serial_layout.setSpacing(5)

        self.baudrate_label = BodyLabel()
        self.baudrate_label.setText('波特率')
        self.baudrate_edit = LineEdit()
        self.baudrate_edit.setText('115200')
        self.baudrate_edit.setFixedWidth(150)
        self.serial_layout.addWidget(self.baudrate_label)
        self.serial_layout.addWidget(self.baudrate_edit)

        self.serial_label = BodyLabel()
        self.serial_label.setText('选择串口')
        self.serial_combo = ComboBox()
        ports_list = self.findSerial()
        self.serial_combo.addItems(ports_list)
        self.serial_combo.setCurrentIndex(len(ports_list) - 1)
        self.serial_combo.setFixedWidth(300)
        self.serial_layout.addWidget(self.serial_label)
        self.serial_layout.addWidget(self.serial_combo)
        self.serial_widget.hide()
        self.tools_layout.addWidget(self.serial_widget)

        # ---- NeuroDance TCP 设置 ----
        self.tcp_widget = QWidget()
        self.tcp_layout = QHBoxLayout(self.tcp_widget)
        self.tcp_layout.setContentsMargins(0, 0, 0, 0)
        self.tcp_layout.setSpacing(5)

        self.tcp_ip_label = BodyLabel()
        self.tcp_ip_label.setText('TCP IP')
        self.tcp_ip_edit = LineEdit()
        self.tcp_ip_edit.setText('10.186.179.92')
        self.tcp_ip_edit.setFixedWidth(150)
        self.tcp_layout.addWidget(self.tcp_ip_label)
        self.tcp_layout.addWidget(self.tcp_ip_edit)

        self.tcp_port_label = BodyLabel()
        self.tcp_port_label.setText('TCP端口')
        self.tcp_port_edit = LineEdit()
        self.tcp_port_edit.setText('8899')
        self.tcp_port_edit.setFixedWidth(100)
        self.tcp_layout.addWidget(self.tcp_port_label)
        self.tcp_layout.addWidget(self.tcp_port_edit)

        self.tools_layout.addWidget(self.tcp_widget)
        self.tcp_widget.show()  # 默认显示 TCP 设置 (NeuroDance TCP)
        
        self.tools_layout.addSpacerItem(QSpacerItem(1, 1, QSizePolicy.Minimum, QSizePolicy.Maximum))

        self.openSerial_button = PushButton()
        self.openSerial_button.setText('连接TCP')
        self.openSerial_button.clicked.connect(self.openSerialEvent)
        self.tools_layout.addWidget(self.openSerial_button)

        self.startRevceiveData_button = PushButton()
        self.startRevceiveData_button.setText('连接')
        self.startRevceiveData_button.clicked.connect(self.receiveDataEvent)
        self.tools_layout.addWidget(self.startRevceiveData_button)

        self.tools_layout.addSpacerItem(QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Maximum))

        self.deviceStateArea = DeviceStateArea()
        self.layout.addWidget(self.deviceStateArea)

        self.eegSignalShow_area = QWidget(objectName='eegSignalShow_area')
        self.eegSignalShow_area.setStyleSheet('#eegSignalShow_area{border-right: 1px solid rgb(239, 239, 239);}')
        self.eegSignalShow_layout = QHBoxLayout(self.eegSignalShow_area)
        self.eegSignalShow_layout.setContentsMargins(0, 0, 0, 0)
        self.eegSignalShow_layout.setSpacing(10)
        self.layout.addWidget(self.eegSignalShow_area)

        self.eegSignalShow_layout_1 = QVBoxLayout()
        self.eegSignalShow_layout_1.setContentsMargins(5, 5, 5, 5)
        self.eegSignalShow_layout_1.setSpacing(10)
        self.eegSignalShow_layout_2 = QVBoxLayout()
        self.eegSignalShow_layout_2.setContentsMargins(5, 5, 5, 5)
        self.eegSignalShow_layout_2.setSpacing(10)

        self.eegSignalShow_layout.addLayout(self.eegSignalShow_layout_1)
        self.eegSignalShow_layout.addLayout(self.eegSignalShow_layout_2)

        # 动态创建通道波形控件（根据当前选中的通道数）
        self.eegSignalShowModules = []
        n_ch = len(getattr(config, 'lsl_selected_channels', list(range(22, 31))))
        self._buildChannelWidgets(n_ch)

        self.layout.addSpacerItem(QSpacerItem(1, 1, QSizePolicy.Maximum, QSizePolicy.Expanding))

        self.deviceStateSignal.connect(self.deviceStateArea.change_device_state)
        if len(self.eegSignalShowModules) > 0:
            self.signalSendSignal.connect(self.eegSignalShowModules[0].set_data)

    def set_car_data_sink(self, sink):
        self.car_data_sink = sink

    def resizeEvent(self, a0):
        super().resizeEvent(a0)
        width = self.width()
        height = self.height()

        self.tools_area.setFixedHeight(height * 0.05)
        self.deviceStateArea.setFixedSize(width, height * 0.3)
        self.eegSignalShow_area.setFixedSize(width * 0.6, height * 0.65)

    def findSerial(self):
        # 查找所有可用串口
        ports_list = list(serial.tools.list_ports.comports())
        ports_list = [f'{list(comport)[0]} {list(comport)[1]}' for comport in ports_list]

        return ports_list

    def openSerialEvent(self):
        device_type = self.device_type_combo.currentIndex()

        if device_type == 0:  # BrainVision RDA
            if self.rda_receiver is None:
                host = self.rda_host_edit.text()
                port = int(self.rda_port_edit.text())
                selected = getattr(config, 'lsl_selected_channels', list(range(32)))
                try:
                    self.rda_receiver = RdaReceiver(
                        host=host,
                        port=port,
                        selected_channels=selected,
                        target_sample_rate=getattr(config, 'lsl_target_sample_rate', 250),
                    )
                    self.rda_receiver.start()
                    w = MessageBox('提示', 'RDA 连接成功！', self)
                    self.openSerial_button.setText('断开RDA')
                except Exception as e:
                    w = MessageBox('提示', f'RDA 连接失败：{str(e)}', self)
                    self.rda_receiver = None
            else:
                if self.is_received_data:
                    self.is_received_data = False
                if self.rda_receiver:
                    self.rda_receiver.close()
                    self.rda_receiver = None
                self.openSerial_button.setText('连接TCP')
                self.startRevceiveData_button.setText('连接')
                w = MessageBox('提示', 'RDA 已断开！', self)

        elif device_type == 1:  # NeuroDance 串口
            if self.ser is None:
                port = self.serial_combo.currentText().split(' ')[0]
                baudrate = int(self.baudrate_edit.text())
                self.ser = serial.Serial(port, baudrate)
                if self.ser.isOpen():
                    w = MessageBox('提示', '打开成功！', self)
                    self.openSerial_button.setText('关闭串口')
                else:
                    w = MessageBox('提示', '打开失败！', self)
                    self.openSerial_button.setText('打开串口')
                    self.ser = None
            else:
                self.closeSerialEvent()
                self.openSerial_button.setText('打开串口')
                self.startRevceiveData_button.setText('连接')
                w = MessageBox('提示', '串口已关闭！', self)
        else:  # TCP设备
            if NdDevice is None:
                w = MessageBox('提示', 'NeuroDance SDK 未安装！', self)
                w.yesButton.setText('确认')
                w.cancelButton.hide()
                w.show()
                return
            if self.nd_device is None:
                tcp_ip = self.tcp_ip_edit.text()
                tcp_port = int(self.tcp_port_edit.text())
                try:
                    self.nd_device = NdDevice(mode='tcp', com='', tcp_ip=tcp_ip, tcp_port=tcp_port)
                    self.nd_device.start()
                    w = MessageBox('提示', 'TCP连接成功！', self)
                    self.openSerial_button.setText('断开TCP')
                except Exception as e:
                    w = MessageBox('提示', f'TCP连接失败：{str(e)}', self)
                    self.nd_device = None
            else:
                if self.is_received_data:
                    self.is_received_data = False
                if self.nd_device:
                    self.nd_device.close()
                    self.nd_device = None
                self.openSerial_button.setText('连接TCP')
                self.startRevceiveData_button.setText('连接')
                w = MessageBox('提示', 'TCP已断开！', self)

        w.yesButton.setText('确认')
        w.cancelButton.hide()
        w.show()

    def closeSerialEvent(self):
        self.is_received_data = False
        with self.lock:
            self.ser.close();
        if self.ser.isOpen():  # 判断串口是否关闭
            print("串口未关闭。")
        else:
            print("串口已关闭。")
            self.ser = None

    def receiveDataEvent(self):
        device_type = self.device_type_combo.currentIndex()

        if device_type == 0:  # BrainVision RDA
            if self.rda_receiver is None:
                return

            if not self.is_received_data:
                self.is_received_data = True
                t = threading.Thread(target=self.receiveRdaDataThread)
                t.start()
                self.startRevceiveData_button.setText('断开')
            else:
                self.is_received_data = False
                self.startRevceiveData_button.setText('连接')

        elif device_type == 1:  # NeuroDance 串口
            if self.ser is None or not self.ser.isOpen():
                return

            if not self.is_received_data:
                self.is_received_data = True
                t = threading.Thread(target=self.receiveDataThread)
                t.start()
                self.startRevceiveData_button.setText('断开')
            else:
                self.closeSerialEvent()
                self.openSerial_button.setText('打开串口')
                self.startRevceiveData_button.setText('连接')
        else:  # NeuroDance TCP
            if self.nd_device is None:
                return

            if not self.is_received_data:
                self.is_received_data = True
                t = threading.Thread(target=self.receiveTcpDataThread)
                t.start()
                self.startRevceiveData_button.setText('断开')
            else:
                self.is_received_data = False
                self.startRevceiveData_button.setText('连接')

    def receiveDataThread(self):
        while self.is_received_data:
            try:
                com_input = self.ser.read(25 * 136)
            except:
                self.is_received_data = False
                self.startRevceiveData_button.setText('连接')
                break
            if com_input:  # 如果读取结果非空，则输出
                # with self.lock:
                #     self.data_buffer += com_input
                with self.lock:
                    self.makeDataWithShape(com_input)

    def makeDataWithShape(self, data):
        if my_dll is None:
            print("makeDataWithShape: LinkMe.dll 不可用，跳过数据解析")
            return
        data_array = (ctypes.c_ubyte * len(data))(*data)
        # 解析数据
        dataSize = my_dll.dataProtocol(data_array, len(data))
        # print(dataSize )
        if dataSize > 0:
            eegData = my_dll.getData();
            data_value = [[eegData[i][j] for j in range(9)] for i in range(dataSize)]
            data_value = np.array(data_value).T
            data_value = data_value[:-1, :]
            raw_data_value = np.asarray(data_value, dtype=float)
            raw_data_value = signal.filtfilt(self.bpB, self.bpA, raw_data_value)  # 50Hz notch
            if paramsController.current_window == 2:
                sink = getattr(self, "car_data_sink", None)
                if callable(sink):
                    sink(raw_data_value)
                else:
                    self.signalSendSignal_car.emit(raw_data_value)
                return
            data_value = signal.filtfilt(self.bpB, self.bpA, data_value)
            data_value = signal.filtfilt(self.bpB2, self.bpA2, data_value)

            if paramsController.current_window == 'home':
                elecValue = my_dll.getElectricityValue()

                flag = [-1, -1, -1, -1, -1, -1, -1, -1]
                fallFlag = (ctypes.c_int * 8)(*flag)
                my_dll.getFallFlag(fallFlag);
                flag = [fallFlag[i] for i in range(len(flag))]

                impedance = [-1, -1, -1, -1, -1, -1, -1, -1]
                impedanceData = (ctypes.c_double * 8)(*impedance)
                my_dll.getImpedance(1000, impedanceData);
                impedanceValue = [impedanceData[i] for i in range(len(impedance))]

                if data_value.shape[-1] == 125:
                    for i in range(min(data_value.shape[0], len(self.eegSignalShowModules))):
                        self.eegSignalShowModules[i].set_data(data_value[i, :])
                        QApplication.processEvents()

                QApplication.processEvents()
                self.deviceStateSignal.emit(elecValue, impedanceValue, flag)
            elif paramsController.current_window == 1:
                self.signalSendSignal_drone.emit(data_value)

                for i in range(len(self.eegSignalShowModules)):
                    self.eegSignalShowModules[i].clearData()
            elif paramsController.current_window == 2:
                sink = getattr(self, "car_data_sink", None)
                if callable(sink):
                    sink(raw_data_value)
                else:
                    self.signalSendSignal_car.emit(raw_data_value)

    def onDeviceTypeChanged(self, index):
        if index == 0:  # BrainVision RDA
            self.lsl_widget.show()
            self.serial_widget.hide()
            self.tcp_widget.hide()
            self.openSerial_button.setText('连接RDA')
        elif index == 1:  # NeuroDance 串口
            self.lsl_widget.hide()
            self.serial_widget.show()
            self.tcp_widget.hide()
            self.openSerial_button.setText('打开串口')
        else:  # NeuroDance TCP
            self.lsl_widget.hide()
            self.serial_widget.hide()
            self.tcp_widget.show()
            self.openSerial_button.setText('连接TCP')
            self.openSerial_button.setText('连接TCP')

    def receiveTcpDataThread(self):
        while self.is_received_data and self.nd_device:
            try:
                read_data = self.nd_device.read_latest_eeg_data(target_freq=250)
                # 处理数据格式 - 从(8, N, 1)格式转换为(8, N)
                if read_data is not None and len(read_data.shape) == 3 and read_data.shape[2] == 1:
                    read_data = read_data.reshape(read_data.shape[0], read_data.shape[1])
                if read_data is not None:
                    if len(read_data.shape) == 3 and read_data.shape[2] == 1:
                        read_data = read_data.reshape(read_data.shape[0], read_data.shape[1])
                    self.processDeviceData(read_data)
                    # 适度限流，避免高频UI投递导致主线程定时器抖动。
                    time.sleep(0.05)
                else:
                    time.sleep(0.05)
                
            except Exception as e:
                print(f"TCP数据接收错误: {str(e)}")
                self.is_received_data = False
                self.startRevceiveData_button.setText('连接')
                break

    def processDeviceData(self, data_value):
        """统一的数据处理入口（BrainVision RDA & NeuroDance TCP 共用）。"""
        if data_value is None or len(data_value) == 0:
            return

        n_channels = data_value.shape[0]

        if paramsController.current_window == 2:
            raw_data_value = np.asarray(data_value, dtype=float)
            raw_data_value = signal.filtfilt(self.bpB, self.bpA, raw_data_value)  # 50Hz notch
            sink = getattr(self, "car_data_sink", None)
            if callable(sink):
                sink(raw_data_value)
            else:
                self.signalSendSignal_car.emit(raw_data_value)
            return

        data_value = signal.filtfilt(self.bpB, self.bpA, data_value)
        data_value = signal.filtfilt(self.bpB2, self.bpA2, data_value)

        if paramsController.current_window == 'home':
            # 动态显示选中通道的波形
            n_display = min(n_channels, len(self.eegSignalShowModules))
            for i in range(n_display):
                self.eegSignalShowModules[i].set_data(data_value[i, :])
                QApplication.processEvents()

            # 计算信号质量用于热力图
            quality = self._computeSignalQuality(data_value)
            elecValue = 100
            impedanceValue = [10] * 8
            flag = [1] * 8

            QApplication.processEvents()
            self.deviceStateSignal.emit(quality, impedanceValue, flag)

        elif paramsController.current_window == 1:
            self.signalSendSignal_drone.emit(data_value)

            for mod in self.eegSignalShowModules:
                mod.clearData()
        elif paramsController.current_window == 2:
            sink = getattr(self, "car_data_sink", None)
            if callable(sink):
                sink(data_value)
            else:
                self.signalSendSignal_car.emit(data_value)
            # 小车模式不刷新设备页波形，避免额外UI开销影响刺激定时。

    # Alias for backward compatibility with NeuroDance path
    processNdDeviceData = processDeviceData

    def receiveRdaDataThread(self):
        """BrainVision RDA 数据接收线程。"""
        while self.is_received_data and self.rda_receiver is not None:
            try:
                read_data = self.rda_receiver.read_latest_eeg_data(target_freq=250)
                if read_data is not None and read_data.size > 0:
                    self.processDeviceData(read_data)
                    time.sleep(0.01)
                else:
                    time.sleep(0.05)
            except Exception as e:
                print(f"RDA数据接收错误: {str(e)}")
                self.is_received_data = False
                self.startRevceiveData_button.setText('连接')
                break
        print("RDA 数据线程已退出")

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _buildChannelWidgets(self, n_channels):
        """清除并重建通道波形控件。通道少时大卡片，通道多时自动缩小。"""
        # 清除旧控件
        for mod in self.eegSignalShowModules:
            mod.setParent(None)
        self.eegSignalShowModules.clear()

        # 清除布局中的旧项
        for layout in (self.eegSignalShow_layout_1, self.eegSignalShow_layout_2):
            while layout.count():
                item = layout.takeAt(0)
                w = item.widget()
                if w:
                    w.setParent(None)

        half = max(1, n_channels // 2 + n_channels % 2)
        # 自适应高度：32 导时 ~40px，9 导时 ~140px
        ch_h = max(30, min(140, 900 // half))
        for i in range(n_channels):
            mod = EEGSignalShowModule(str(i + 1))
            mod.setFixedHeight(ch_h)
            self.eegSignalShowModules.append(mod)
            if i < half:
                self.eegSignalShow_layout_1.addWidget(mod)
            else:
                self.eegSignalShow_layout_2.addWidget(mod)

    def _showChannelSelectionDialog(self):
        """弹出32导通道选择对话框。"""
        dlg = QDialog(self)
        dlg.setWindowTitle('选择 EEG 通道（0-based 索引）')
        dlg.resize(600, 400)
        layout = QVBoxLayout(dlg)

        lbl = BodyLabel()
        lbl.setText('勾选要使用的通道（默认选中枕叶区 22-30）。选中后将重建波形显示。')
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        grid = QGridLayout(container)
        grid.setSpacing(4)

        selected = getattr(config, 'lsl_selected_channels', list(range(22, 31)))
        checkboxes = []
        cols = 8
        for ch in range(32):
            cb = QCheckBox(str(ch))
            cb.setChecked(ch in selected)
            grid.addWidget(cb, ch // cols, ch % cols)
            checkboxes.append(cb)

        scroll.setWidget(container)
        layout.addWidget(scroll)

        btn_layout = QHBoxLayout()
        btn_all = PushButton()
        btn_all.setText('全选')
        btn_none = PushButton()
        btn_none.setText('全不选')
        btn_occipital = PushButton()
        btn_occipital.setText('枕叶默认 (22-30)')
        btn_ok = PushButton()
        btn_ok.setText('确定')
        btn_cancel = PushButton()
        btn_cancel.setText('取消')

        def select_range(start, end):
            for i, cb in enumerate(checkboxes):
                cb.setChecked(start <= i <= end)

        btn_all.clicked.connect(lambda: select_range(0, 31))
        btn_none.clicked.connect(lambda: [cb.setChecked(False) for cb in checkboxes])
        btn_occipital.clicked.connect(lambda: select_range(22, 30))
        btn_ok.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)

        btn_layout.addWidget(btn_all)
        btn_layout.addWidget(btn_none)
        btn_layout.addWidget(btn_occipital)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

        if dlg.exec_() == QDialog.Accepted:
            new_selected = [i for i, cb in enumerate(checkboxes) if cb.isChecked()]
            if len(new_selected) == 0:
                w = MessageBox('提示', '至少选择1个通道！', self)
                w.yesButton.setText('确认')
                w.cancelButton.hide()
                w.show()
                return
            config.change('lsl_selected_channels', new_selected)
            self.lsl_channel_count_label.setText(f'已选 {len(new_selected)}/32 导')
            self._buildChannelWidgets(len(new_selected))
            # 重建信号连接
            if len(self.eegSignalShowModules) > 0:
                try:
                    self.signalSendSignal.disconnect()
                except Exception:
                    pass
                self.signalSendSignal.connect(self.eegSignalShowModules[0].set_data)

    def _computeSignalQuality(self, data):
        """计算每个通道的信号质量指标（用于热力图显示）。

        Returns
        -------
        list[dict]: 每通道 {'variance', 'noise_50hz_ratio', 'is_saturated'}
        """
        n_ch = data.shape[0]
        metrics = []
        for i in range(n_ch):
            ch_data = np.asarray(data[i, :], dtype=float).ravel()
            if len(ch_data) < 16:
                metrics.append({'variance': 0.0, 'noise_50hz_ratio': 0.0, 'is_saturated': 0.0})
                continue

            variance = float(np.var(ch_data))

            # 50Hz 工频噪声比例
            try:
                f, Pxx = signal.welch(
                    ch_data, fs=250, nperseg=min(256, len(ch_data)),
                    detrend='constant'
                )
                band_mask = (f >= 48) & (f <= 52)
                noise_50hz_ratio = float(
                    np.sum(Pxx[band_mask]) / max(np.sum(Pxx), 1e-12)
                )
            except Exception:
                noise_50hz_ratio = 0.0

            # 饱和检测
            peak = float(np.max(np.abs(ch_data)))
            is_saturated = 1.0 if peak > 500.0 else 0.0

            metrics.append({
                'variance': variance,
                'noise_50hz_ratio': noise_50hz_ratio,
                'is_saturated': is_saturated,
            })
        return metrics
