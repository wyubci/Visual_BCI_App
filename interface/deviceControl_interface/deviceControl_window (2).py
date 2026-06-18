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
from scipy import signal

from .deviceState_area import DeviceStateArea
from .signalShow_window import EEGSignalShowModule
from controller.params_controller import paramsController
from .nd_device_demo import NdDevice

my_dll = ctypes.CDLL("source/device/LinkMe.dll")

# init functions of dll
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
    signalSendSignal_keyboard = pyqtSignal(object)
    signalSendSignal_care = pyqtSignal(object)
    signalSendSignal_test = pyqtSignal(object)
    signalSendSignal_dualFreq = pyqtSignal(object)
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
        self.device_type_combo.addItems(['串口设备', 'TCP设备'])
        self.device_type_combo.setCurrentIndex(1)  # 默认选择TCP设备
        self.device_type_combo.currentIndexChanged.connect(self.onDeviceTypeChanged)
        self.tools_layout.addWidget(self.device_type_label)
        self.tools_layout.addWidget(self.device_type_combo)

        self.tools_layout.addSpacing(10)

        # 串口设置
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
        self.serial_widget.hide()  # 默认隐藏串口设置
        self.tools_layout.addWidget(self.serial_widget)

        # TCP设置
        self.tcp_widget = QWidget()
        self.tcp_layout = QHBoxLayout(self.tcp_widget)
        self.tcp_layout.setContentsMargins(0, 0, 0, 0)
        self.tcp_layout.setSpacing(5)
        
        self.tcp_ip_label = BodyLabel()
        self.tcp_ip_label.setText('TCP IP')
        self.tcp_ip_edit = LineEdit()
        self.tcp_ip_edit.setText('192.168.1.27')
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
        self.tcp_widget.show()  # 默认显示TCP设置
        
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
        # Removed hardcoded style to allow QSS theming
        # self.eegSignalShow_area.setStyleSheet('#eegSignalShow_area{border-right: 1px solid rgb(239, 239, 239);}')
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


        self.eegSignalShowModule_1 = EEGSignalShowModule('1')
        self.eegSignalShowModule_2 = EEGSignalShowModule('2')
        self.eegSignalShowModule_3 = EEGSignalShowModule('3')
        self.eegSignalShowModule_4 = EEGSignalShowModule('4')
        self.eegSignalShowModule_5 = EEGSignalShowModule('5')
        self.eegSignalShowModule_6 = EEGSignalShowModule('6')
        self.eegSignalShowModule_7 = EEGSignalShowModule('7')
        self.eegSignalShowModule_8 = EEGSignalShowModule('8')
        self.eegSignalShowModule_1.setFixedHeight(140)
        self.eegSignalShowModule_2.setFixedHeight(140)
        self.eegSignalShowModule_3.setFixedHeight(140)
        self.eegSignalShowModule_4.setFixedHeight(140)
        self.eegSignalShowModule_5.setFixedHeight(140)
        self.eegSignalShowModule_6.setFixedHeight(140)
        self.eegSignalShowModule_7.setFixedHeight(140)
        self.eegSignalShowModule_8.setFixedHeight(140)

        self.eegSignalShow_layout_1.addWidget(self.eegSignalShowModule_1)
        self.eegSignalShow_layout_1.addWidget(self.eegSignalShowModule_2)
        self.eegSignalShow_layout_1.addWidget(self.eegSignalShowModule_3)
        self.eegSignalShow_layout_1.addWidget(self.eegSignalShowModule_4)
        self.eegSignalShow_layout_2.addWidget(self.eegSignalShowModule_5)
        self.eegSignalShow_layout_2.addWidget(self.eegSignalShowModule_6)
        self.eegSignalShow_layout_2.addWidget(self.eegSignalShowModule_7)
        self.eegSignalShow_layout_2.addWidget(self.eegSignalShowModule_8)

        self.layout.addSpacerItem(QSpacerItem(1, 1, QSizePolicy.Maximum, QSizePolicy.Expanding))

        self.deviceStateSignal.connect(self.deviceStateArea.change_device_state)
        self.signalSendSignal.connect(self.eegSignalShowModule_1.set_data)

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
        
        if device_type == 0:  # 串口设备
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
        
        if device_type == 0:  # 串口设备
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
        else:  # TCP设备
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
        data_array = (ctypes.c_ubyte * len(data))(*data)
        # 解析数据
        dataSize = my_dll.dataProtocol(data_array, len(data))
        # print(dataSize )
        if dataSize > 0:
            eegData = my_dll.getData();
            data_value = [[eegData[i][j] for j in range(9)] for i in range(dataSize)]
            data_value = np.array(data_value).T
            data_value = data_value[:-1, :]
            # data_value = signal.filtfilt(self.bpB, self.bpA, data_value)
            # data_value = signal.filtfilt(self.bpB2, self.bpA2, data_value)

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
                    for i in range(data_value.shape[0]):
                        attr = getattr(self, f'eegSignalShowModule_{i + 1}')
                        attr.set_data(data_value[i, :])
                        QApplication.processEvents()

                QApplication.processEvents()
                self.deviceStateSignal.emit(elecValue, impedanceValue, flag)
            elif paramsController.current_window == 1:
                self.signalSendSignal_test.emit(data_value)

                for i in range(8):
                    attr = getattr(self, f'eegSignalShowModule_{i + 1}')
                    attr.clearData()
            elif paramsController.current_window == 2:
                self.signalSendSignal_keyboard.emit(data_value)

                for i in range(8):
                    attr = getattr(self, f'eegSignalShowModule_{i + 1}')
                    attr.clearData()
            elif paramsController.current_window == 3:
                self.signalSendSignal_care.emit(data_value)

                for i in range(8):
                    attr = getattr(self, f'eegSignalShowModule_{i + 1}')
                    attr.clearData()
            elif paramsController.current_window == 4:
                self.signalSendSignal_dualFreq.emit(data_value)

                for i in range(8):
                    attr = getattr(self, f'eegSignalShowModule_{i + 1}')
                    attr.clearData()
            elif paramsController.current_window == 5:  # 假设5是脑控Drone的索引
                self.signalSendSignal_drone.emit(data_value)
                
                for i in range(8):
                    attr = getattr(self, f'eegSignalShowModule_{i + 1}')
                    attr.clearData()
            elif paramsController.current_window == 6:  # 假设6是脑控小车的索引
                self.signalSendSignal_car.emit(data_value)
                
                for i in range(8):
                    attr = getattr(self, f'eegSignalShowModule_{i + 1}')
                    attr.clearData()

    def onDeviceTypeChanged(self, index):
        if index == 0:  # 串口设备
            self.serial_widget.show()
            self.tcp_widget.hide()
            self.openSerial_button.setText('打开串口')
        else:  # TCP设备
            self.serial_widget.hide()
            self.tcp_widget.show()
            self.openSerial_button.setText('连接TCP')

    def receiveTcpDataThread(self):
        while self.is_received_data and self.nd_device:
            try:
                millis_second = int(round(time.time() * 1000))
                time_span = 1000
                read_data = self.nd_device.read_latest_eeg_data()
                # 处理数据格式 - 从(8, N, 1)格式转换为(8, N)
                if read_data is not None and len(read_data.shape) == 3 and read_data.shape[2] == 1:
                    read_data = read_data.reshape(read_data.shape[0], read_data.shape[1])
                if read_data is not None:
                    if len(read_data.shape) == 3 and read_data.shape[2] == 1:
                        read_data = read_data.reshape(read_data.shape[0], read_data.shape[1])
                    self.processNdDeviceData(read_data)
                
                time.sleep(0.15)
                
            except Exception as e:
                print(f"TCP数据接收错误: {str(e)}")
                self.is_received_data = False
                self.startRevceiveData_button.setText('连接')
                break

    def processNdDeviceData(self, data_value):
        if data_value is None or len(data_value) == 0:
            return
            
        if paramsController.current_window == 'home':
            # 在主界面显示波形
            if data_value.shape[0] == 8:  # 确保有8个通道
                for i in range(data_value.shape[0]):
                    attr = getattr(self, f'eegSignalShowModule_{i + 1}')
                    attr.set_data(data_value[i, :])
                    QApplication.processEvents()
                
                # 模拟设备状态信息
                elecValue = 100  # 电量值
                impedanceValue = [10, 10, 10, 10, 10, 10, 10, 10]  # 阻抗值
                flag = [1, 1, 1, 1, 1, 1, 1, 1]  # 连接状态
                
                QApplication.processEvents()
                self.deviceStateSignal.emit(elecValue, impedanceValue, flag)
                
            else:
                print(f"通道数量不匹配: {data_value.shape}")
                
        elif paramsController.current_window == 1:
            self.signalSendSignal_test.emit(data_value)
            
            for i in range(8):
                attr = getattr(self, f'eegSignalShowModule_{i + 1}')
                attr.clearData()
        elif paramsController.current_window == 2:
            self.signalSendSignal_keyboard.emit(data_value)
            
            for i in range(8):
                attr = getattr(self, f'eegSignalShowModule_{i + 1}')
                attr.clearData()
        elif paramsController.current_window == 3:
            self.signalSendSignal_care.emit(data_value)
            
            for i in range(8):
                attr = getattr(self, f'eegSignalShowModule_{i + 1}')
                attr.clearData()
        elif paramsController.current_window == 4:
            self.signalSendSignal_dualFreq.emit(data_value)
            
            for i in range(8):
                attr = getattr(self, f'eegSignalShowModule_{i + 1}')
                attr.clearData()
        elif paramsController.current_window == 5:  # 假设5是脑控Drone的索引
            self.signalSendSignal_drone.emit(data_value)
            
            for i in range(8):
                attr = getattr(self, f'eegSignalShowModule_{i + 1}')
                attr.clearData()
        elif paramsController.current_window == 6:  # 假设6是脑控小车的索引
            self.signalSendSignal_car.emit(data_value)
            
            for i in range(8):
                attr = getattr(self, f'eegSignalShowModule_{i + 1}')
                attr.clearData()